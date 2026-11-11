from __future__ import annotations

import asyncio
import gc
import time
import weakref

import pytest
from conftest import CountingLoader, EchoModel, spec

from lazyserve import LoadFailed, MemoryBusy, ModelTooLarge, UnknownModel
from lazyserve.registry import InvalidInput, PredictFailed


def loaded_names(registry) -> list[str]:
    return sorted(r.name for r in registry.records() if r.loaded)


def test_nothing_loads_until_first_request(make_registry):
    loader = CountingLoader("a")
    registry = make_registry([spec("a", loader)])
    assert loader.calls == 0
    assert registry.status("a")["status"] == "unloaded"

    result = asyncio.run(registry.predict("a", [1, 2]))

    assert result.outputs == {"tag": "a", "inputs": [1, 2]}
    assert result.cold_start is True
    assert loader.calls == 1
    assert registry.status("a")["status"] == "loaded"


def test_second_request_is_warm(make_registry):
    registry = make_registry([spec("a")])

    async def run():
        first = await registry.predict("a", 1)
        second = await registry.predict("a", 2)
        return first, second

    first, second = asyncio.run(run())
    assert first.cold_start is True
    assert second.cold_start is False
    assert registry.status("a")["hits"] == 2


def test_single_flight_twenty_concurrent_requests_one_load(make_registry):
    loader = CountingLoader("slow", delay_s=0.2)
    registry = make_registry([spec("slow", loader)])

    async def run():
        return await asyncio.gather(*(registry.predict("slow", i) for i in range(20)))

    results = asyncio.run(run())

    assert loader.calls == 1
    assert registry.record("slow").loads == 1
    assert [r.outputs["inputs"] for r in results] == list(range(20))
    assert all(r.cold_start for r in results)
    assert registry.status("slow")["hits"] == 20


def test_lru_eviction_order(make_registry, clock):
    closed: list[str] = []
    specs = [
        spec(n, CountingLoader(n, closed=closed), memory_mb=100) for n in ("a", "b", "c", "d", "e")
    ]
    registry = make_registry(specs, max_memory_mb=300)

    async def use(name):
        clock.advance(1)
        await registry.predict(name, None)

    async def run():
        for name in ("a", "b", "c"):
            await use(name)
        await use("a")  # a is now the most recently used; b is the oldest
        await use("d")  # needs 100 MB: evicts b
        await use("e")  # evicts c
        await use("b")  # evicts a (d and e are newer)

    asyncio.run(run())

    assert closed == ["b", "c", "a"]
    assert loaded_names(registry) == ["b", "d", "e"]
    assert registry.memory_used_mb() == 300
    assert registry.record("b").unloads["lru"] == 1
    assert registry.record("b").loads == 2


def test_eviction_frees_as_few_models_as_needed(make_registry, clock):
    specs = [
        spec("s1", memory_mb=100),
        spec("s2", memory_mb=100),
        spec("s3", memory_mb=100),
        spec("big", memory_mb=200),
    ]
    registry = make_registry(specs, max_memory_mb=300)

    async def run():
        for name in ("s1", "s2", "s3", "big"):
            clock.advance(1)
            await registry.predict(name, None)

    asyncio.run(run())
    assert loaded_names(registry) == ["big", "s3"]


def test_pinned_models_are_never_evicted(make_registry, clock):
    specs = [
        spec("pinned", memory_mb=100, pin=True),
        spec("a", memory_mb=100),
        spec("b", memory_mb=100),
        spec("c", memory_mb=100),
    ]
    registry = make_registry(specs, max_memory_mb=300)

    async def run():
        for name in ("pinned", "a", "b", "c", "a", "b", "c"):
            clock.advance(1)
            await registry.predict(name, None)

    asyncio.run(run())

    # pinned is the least recently used every time, yet it stays loaded.
    assert registry.record("pinned").loaded
    assert registry.record("pinned").unloads.total() == 0
    assert registry.record("a").unloads["lru"] >= 1


def test_model_bigger_than_budget_is_rejected_without_loading(make_registry):
    loader = CountingLoader("huge")
    registry = make_registry([spec("huge", loader, memory_mb=1500)], max_memory_mb=1000)

    with pytest.raises(ModelTooLarge) as info:
        asyncio.run(registry.predict("huge", None))

    assert info.value.status_code == 507
    assert "needs 1500 MB but the whole budget is 1000 MB" in info.value.message
    assert loader.calls == 0
    assert registry.status("huge")["status"] == "unloaded"


def test_model_that_only_fits_without_pinned_models_is_rejected(make_registry, clock):
    specs = [spec("pinned", memory_mb=600, pin=True), spec("big", memory_mb=500)]
    registry = make_registry(specs, max_memory_mb=1000)

    async def run():
        await registry.predict("pinned", None)
        await registry.predict("big", None)

    with pytest.raises(ModelTooLarge) as info:
        asyncio.run(run())
    assert "only 400 MB" in info.value.message
    assert "pinned" in info.value.message
    assert registry.record("pinned").loaded


class SlowModel(EchoModel):
    def predict(self, inputs):
        time.sleep(0.3)
        return super().predict(inputs)


def test_busy_models_are_not_evicted(make_registry):
    specs = [
        spec("busy", lambda: SlowModel("busy"), memory_mb=100),
        spec("other", memory_mb=100),
    ]
    registry = make_registry(specs, max_memory_mb=150)

    async def run():
        task = asyncio.create_task(registry.predict("busy", None))
        while registry.record("busy").in_use == 0:
            await asyncio.sleep(0.01)
        with pytest.raises(MemoryBusy) as info:
            await registry.predict("other", None)
        await task
        # Once the busy model is idle again, it can be evicted.
        await registry.predict("other", None)
        return info.value

    err = asyncio.run(run())
    assert err.status_code == 503
    assert "Retry shortly" in err.message
    assert loaded_names(registry) == ["other"]


def test_idle_ttl_unload_with_controllable_clock(make_registry, clock):
    specs = [
        spec("short", memory_mb=10, idle_ttl_s=30),
        spec("forever", memory_mb=10),
        spec("pinned", memory_mb=10, idle_ttl_s=1, pin=True),
    ]
    registry = make_registry(specs)

    async def run():
        for name in ("short", "forever", "pinned"):
            await registry.predict(name, None)
        clock.advance(20)
        first = await registry.sweep_idle()
        clock.advance(11)
        second = await registry.sweep_idle()
        result = await registry.predict("short", None)
        return first, second, result

    first, second, result = asyncio.run(run())

    assert first == []
    assert second == ["short"]
    assert registry.record("short").unloads["idle"] == 1
    assert registry.record("forever").loaded
    assert registry.record("pinned").loaded
    assert result.cold_start is True
    assert registry.record("short").loads == 2


def test_recent_use_resets_idle_timer(make_registry, clock):
    registry = make_registry([spec("m", idle_ttl_s=30)])

    async def run():
        await registry.predict("m", None)
        clock.advance(25)
        await registry.predict("m", None)
        clock.advance(25)
        return await registry.sweep_idle()

    assert asyncio.run(run()) == []
    assert registry.record("m").loaded


def test_background_sweeper_unloads_idle_models(make_registry, clock):
    registry = make_registry([spec("m", idle_ttl_s=5)], idle_check_interval_s=0.02)

    async def run():
        await registry.start()
        try:
            await registry.predict("m", None)
            clock.advance(10)
            for _ in range(100):
                if not registry.record("m").loaded:
                    break
                await asyncio.sleep(0.02)
        finally:
            await registry.stop()

    asyncio.run(run())
    assert registry.record("m").unloads["idle"] == 1


def test_loader_exception_does_not_poison_registry(make_registry):
    loader = CountingLoader("flaky", fail_times=1)
    registry = make_registry([spec("flaky", loader)])

    with pytest.raises(LoadFailed) as info:
        asyncio.run(registry.predict("flaky", None))
    assert "not reachable" in info.value.message
    status = registry.status("flaky")
    assert status["status"] == "unloaded"
    assert status["load_failures"] == 1
    assert "OSError" in status["last_error"]

    result = asyncio.run(registry.predict("flaky", "retry"))
    assert result.outputs["inputs"] == "retry"
    assert loader.calls == 2
    assert registry.status("flaky")["last_error"] is None


def test_concurrent_waiters_all_see_the_failure_then_retry_works(make_registry):
    loader = CountingLoader("flaky", delay_s=0.1, fail_times=1)
    registry = make_registry([spec("flaky", loader)])

    async def run():
        first = await asyncio.gather(
            *(registry.predict("flaky", i) for i in range(5)), return_exceptions=True
        )
        second = await registry.predict("flaky", "again")
        return first, second

    first, second = asyncio.run(run())
    assert all(isinstance(r, LoadFailed) for r in first)
    assert loader.calls == 2
    assert second.outputs["inputs"] == "again"


def test_loader_returning_object_without_predict_fails_cleanly(make_registry):
    registry = make_registry([spec("bad", lambda: object())])
    with pytest.raises(LoadFailed, match="no predict"):
        asyncio.run(registry.predict("bad", None))
    assert registry.status("bad")["status"] == "unloaded"


def test_import_path_loader_and_options(make_registry):
    registry = make_registry(
        [
            spec(
                "clf",
                "lazyserve.examples.logistic:load",
                options={"n_samples": 100, "epochs": 50},
            )
        ]
    )
    result = asyncio.run(registry.predict("clf", [[2.0, 2.0], [-2.0, -2.0]]))
    assert result.outputs["labels"] == [1, 0]


def test_missing_import_path_is_a_load_failure(make_registry):
    registry = make_registry([spec("ghost", "no_such_package.models:load")])
    with pytest.raises(LoadFailed, match="ModuleNotFoundError"):
        asyncio.run(registry.predict("ghost", None))


def test_measured_rss_delta_is_used_when_no_estimate(make_registry, clock):
    rss = {"value": 500 * 1024 * 1024}

    def fake_rss():
        return rss["value"]

    def grows_by(mb):
        def loader():
            rss["value"] += mb * 1024 * 1024
            return EchoModel("m")

        return loader

    specs = [spec("a", grows_by(60)), spec("b", grows_by(70))]
    registry = make_registry(specs, max_memory_mb=100, rss=fake_rss)

    async def run():
        await registry.predict("a", None)
        clock.advance(1)
        await registry.predict("b", None)

    asyncio.run(run())

    a, b = registry.status("a"), registry.status("b")
    assert a["measured_mb"] == 60 and a["memory_source"] == "measured"
    # b's size was unknown until it loaded; then a had to go to get back under budget.
    assert b["status"] == "loaded" and b["memory_mb"] == 70
    assert a["status"] == "unloaded" and a["unloads"]["lru"] == 1


def test_declared_estimate_wins_over_measurement(make_registry):
    registry = make_registry([spec("a", memory_mb=42)])
    asyncio.run(registry.predict("a", None))
    status = registry.status("a")
    assert status["memory_mb"] == 42
    assert status["memory_source"] == "declared"
    assert status["measured_mb"] == 0  # constant fake RSS


def test_unload_drops_the_last_reference(make_registry):
    holder = {}

    def loader():
        model = EchoModel("w")
        holder["ref"] = weakref.ref(model)
        return model

    registry = make_registry([spec("w", loader)])

    async def run():
        await registry.predict("w", None)
        assert holder["ref"]() is not None
        assert await registry.unload("w") is True
        assert await registry.unload("w") is False

    asyncio.run(run())
    gc.collect()
    assert holder["ref"]() is None
    assert registry.record("w").unloads["manual"] == 1


def test_unknown_model(make_registry):
    registry = make_registry([spec("a")])
    with pytest.raises(UnknownModel):
        asyncio.run(registry.predict("nope", None))


def test_predict_errors_are_classified(make_registry):
    registry = make_registry([spec("a")])
    with pytest.raises(InvalidInput):
        asyncio.run(registry.predict("a", "bad"))
    with pytest.raises(PredictFailed, match="model crashed"):
        asyncio.run(registry.predict("a", "boom"))
    assert registry.record("a").in_use == 0


def test_warm_list_loads_at_start_and_tolerates_failures(make_registry):
    good, bad = CountingLoader("good"), CountingLoader("bad", fail_times=99)
    registry = make_registry([spec("good", good), spec("bad", bad)], warm=["good", "bad"])

    async def run():
        await registry.start()
        await registry.stop()

    asyncio.run(run())
    assert registry.record("good").loaded
    assert not registry.record("bad").loaded
    assert registry.record("bad").load_failures == 1


def test_ballast_example_holds_and_releases_memory(make_registry):
    registry = make_registry(
        [spec("ballast", "lazyserve.examples.ballast:load", options={"size_mb": 2})]
    )

    async def run():
        result = await registry.predict("ballast", "hi")
        model = registry.record("ballast").model
        await registry.unload("ballast")
        return result, model

    result, model = asyncio.run(run())
    assert result.outputs["echo"] == "hi" and result.outputs["ballast_mb"] == 2
    assert model._block is None  # close() unmapped the block
