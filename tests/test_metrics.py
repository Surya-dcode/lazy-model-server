from __future__ import annotations

import asyncio
import re

from conftest import spec

from lazyserve.metrics import escape_label, render_metrics

SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r'(?:\{(?P<labels>[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\.)*"'
    r'(?:,[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\.)*")*)\})?'
    r" (?P<value>-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$"
)


def parse(text: str) -> dict[str, float]:
    """Check the exposition format and return {'name{labels}': value}."""
    assert text.endswith("\n")
    samples: dict[str, float] = {}
    families: dict[str, str] = {}
    helped: set[str] = set()
    for line in text.splitlines():
        if line.startswith("# HELP "):
            helped.add(line.split()[2])
            continue
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            assert kind in {"counter", "gauge", "summary", "histogram", "untyped"}
            assert name in helped, f"TYPE before HELP for {name}"
            families[name] = kind
            continue
        m = SAMPLE_RE.match(line)
        assert m, f"bad sample line: {line!r}"
        name = m["name"]
        family = re.sub(r"_(sum|count)$", "", name) if name not in families else name
        assert family in families, f"sample {name} has no TYPE line"
        key = f"{name}{{{m['labels']}}}" if m["labels"] else name
        assert key not in samples, f"duplicate sample {key}"
        samples[key] = float(m["value"])
    return samples


def test_metrics_format_and_values(make_registry, clock):
    specs = [spec("a", memory_mb=100), spec("b", memory_mb=100), spec("c", memory_mb=50)]
    registry = make_registry(specs, max_memory_mb=150)

    async def run():
        await registry.predict("a", None)
        clock.advance(1)
        await registry.predict("a", None)
        clock.advance(1)
        await registry.predict("b", None)  # evicts a
        clock.advance(1)
        await registry.predict("c", None)
        await registry.unload("c")

    asyncio.run(run())
    samples = parse(render_metrics(registry))

    assert samples['lazyserve_model_loads_total{model="a"}'] == 1
    assert samples['lazyserve_model_unloads_total{model="a",reason="lru"}'] == 1
    assert samples['lazyserve_model_unloads_total{model="c",reason="manual"}'] == 1
    assert samples['lazyserve_model_requests_total{model="a"}'] == 2
    assert samples['lazyserve_model_loaded{model="a"}'] == 0
    assert samples['lazyserve_model_loaded{model="b"}'] == 1
    assert samples['lazyserve_model_memory_mb{model="b"}'] == 100
    assert samples['lazyserve_model_load_seconds_count{model="b"}'] == 1
    assert samples['lazyserve_model_load_seconds_sum{model="b"}'] >= 0
    assert samples["lazyserve_memory_used_mb"] == 100
    assert samples["lazyserve_memory_budget_mb"] == 150
    assert samples["lazyserve_process_rss_bytes"] == 100 * 1024 * 1024


def test_no_budget_omits_budget_gauge(make_registry):
    samples = parse(render_metrics(make_registry([spec("a")])))
    assert "lazyserve_memory_budget_mb" not in samples
    assert samples['lazyserve_model_loaded{model="a"}'] == 0


def test_label_escaping():
    assert escape_label('a"b\\c\nd') == 'a\\"b\\\\c\\nd'
