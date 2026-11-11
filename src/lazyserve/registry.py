"""The model registry: lazy loading, single-flight, memory budget and idle unload."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .config import ModelSpec, ServerConfig
from .memory import MB, process_rss_bytes, release_memory

log = logging.getLogger("lazyserve")

UNLOAD_REASONS = ("lru", "idle", "manual")


class RegistryError(Exception):
    """Base class for errors the HTTP layer turns into responses."""

    status_code = 500
    code = "error"

    def __init__(self, model: str, message: str) -> None:
        super().__init__(message)
        self.model = model
        self.message = message


class UnknownModel(RegistryError):
    status_code = 404
    code = "unknown_model"


class ModelTooLarge(RegistryError):
    """The model can never fit: it is bigger than the budget left after pinned models."""

    status_code = 507
    code = "model_too_large"


class MemoryBusy(RegistryError):
    """Not enough memory right now because the models that could be evicted are in use."""

    status_code = 503
    code = "memory_busy"


class LoadFailed(RegistryError):
    status_code = 500
    code = "load_failed"


class InvalidInput(RegistryError):
    status_code = 422
    code = "invalid_input"


class PredictFailed(RegistryError):
    status_code = 500
    code = "predict_failed"


@dataclass
class ModelRecord:
    """Runtime state for one configured model."""

    spec: ModelSpec
    model: Any = None
    status: str = "unloaded"
    measured_mb: float | None = None
    load_seconds: float | None = None
    last_used: float | None = None
    hits: int = 0
    in_use: int = 0
    loads: int = 0
    load_failures: int = 0
    load_seconds_total: float = 0.0
    unloads: Counter = field(default_factory=Counter)
    last_error: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def size_mb(self) -> float:
        """Memory charged against the budget: the declared estimate, else the measured delta."""
        if self.spec.memory_mb is not None:
            return self.spec.memory_mb
        return self.measured_mb or 0.0

    @property
    def size_source(self) -> str | None:
        if self.spec.memory_mb is not None:
            return "declared"
        if self.measured_mb is not None:
            return "measured"
        return None


@dataclass
class PredictResult:
    outputs: Any
    cold_start: bool


class ModelRegistry:
    """Holds every configured model and decides when each one is in memory.

    All state changes happen on the event loop. Loader functions and
    ``predict`` calls run in worker threads so they do not block it.

    Loads are serialized by one lock. Two different models never load at the
    same time, which keeps the RSS delta of a load attributable to that model
    and keeps the budget arithmetic simple.
    """

    def __init__(
        self,
        models: Iterable[ModelSpec],
        *,
        max_memory_mb: float | None = None,
        idle_check_interval_s: float = 10.0,
        warm: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
        rss: Callable[[], int] = process_rss_bytes,
    ) -> None:
        self._records = {spec.name: ModelRecord(spec) for spec in models}
        self.max_memory_mb = max_memory_mb
        self.idle_check_interval_s = idle_check_interval_s
        self.warm = list(warm)
        self._clock = clock
        self._rss = rss
        self._lock = asyncio.Lock()
        self._loading: dict[str, asyncio.Task] = {}
        self._sweeper: asyncio.Task | None = None

    @classmethod
    def from_config(cls, config: ServerConfig, **kwargs: Any) -> ModelRegistry:
        return cls(
            config.models,
            max_memory_mb=config.max_memory_mb,
            idle_check_interval_s=config.idle_check_interval_s,
            warm=config.warm,
            **kwargs,
        )

    # ----- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Load the warm list and start the idle sweeper."""
        for name in self.warm:
            try:
                await self.load(name)
                log.info("warm load done: %s", name)
            except RegistryError as exc:
                log.error("warm load failed for %s: %s", name, exc.message)
        if any(r.spec.idle_ttl_s for r in self._records.values()):
            self._sweeper = asyncio.create_task(self._sweep_loop(), name="lazyserve-idle-sweeper")

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None

    # ----- public operations ----------------------------------------------

    async def predict(self, name: str, inputs: Any) -> PredictResult:
        """Run ``predict`` on a model, loading it first if needed."""
        model, cold = await self._acquire(name)
        rec = self._records[name]
        rec.hits += 1
        try:
            outputs = await asyncio.to_thread(model.predict, inputs)
        except (ValueError, TypeError) as exc:
            raise InvalidInput(name, f"model {name!r} rejected the inputs: {exc}") from exc
        except Exception as exc:
            raise PredictFailed(
                name, f"model {name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            self._release(rec)
        return PredictResult(outputs=outputs, cold_start=cold)

    async def load(self, name: str) -> bool:
        """Make sure a model is loaded. Returns True if this call had to wait for a load."""
        _, cold = await self._acquire(name)
        self._release(self._records[name])
        return cold

    async def unload(self, name: str, reason: str = "manual") -> bool:
        """Unload a model. Returns False if it was not loaded."""
        rec = self._get(name)
        async with self._lock:
            if not rec.loaded:
                return False
            self._unload_locked(rec, reason)
            return True

    async def sweep_idle(self) -> list[str]:
        """Unload unpinned models that have been idle longer than their TTL."""
        unloaded = []
        async with self._lock:
            now = self._clock()
            for rec in self._records.values():
                ttl = rec.spec.idle_ttl_s
                if not rec.loaded or rec.spec.pin or ttl is None or rec.in_use:
                    continue
                if rec.last_used is not None and now - rec.last_used > ttl:
                    self._unload_locked(rec, "idle")
                    unloaded.append(rec.name)
        return unloaded

    # ----- introspection ---------------------------------------------------

    def records(self) -> list[ModelRecord]:
        return list(self._records.values())

    def record(self, name: str) -> ModelRecord:
        return self._get(name)

    def memory_used_mb(self) -> float:
        return sum(r.size_mb for r in self._records.values() if r.loaded)

    def rss_bytes(self) -> int:
        return self._rss()

    def status(self, name: str) -> dict[str, Any]:
        rec = self._get(name)
        now = self._clock()
        return {
            "name": rec.name,
            "status": rec.status,
            "pinned": rec.spec.pin,
            "idle_ttl_s": rec.spec.idle_ttl_s,
            "memory_mb": round(rec.size_mb, 2) if rec.size_source else None,
            "memory_source": rec.size_source,
            "declared_mb": rec.spec.memory_mb,
            "measured_mb": _round(rec.measured_mb),
            "load_seconds": _round(rec.load_seconds, 4),
            "last_used_s_ago": _round(now - rec.last_used if rec.last_used is not None else None),
            "hits": rec.hits,
            "in_use": rec.in_use,
            "loads": rec.loads,
            "load_failures": rec.load_failures,
            "unloads": {reason: rec.unloads[reason] for reason in UNLOAD_REASONS},
            "last_error": rec.last_error,
        }

    def statuses(self) -> list[dict[str, Any]]:
        return [self.status(name) for name in self._records]

    # ----- internals -------------------------------------------------------

    def _get(self, name: str) -> ModelRecord:
        try:
            return self._records[name]
        except KeyError:
            raise UnknownModel(name, f"no model named {name!r} is configured") from None

    async def _acquire(self, name: str) -> tuple[Any, bool]:
        """Return a loaded model and mark it in use. Concurrent callers share one load."""
        rec = self._get(name)
        cold = False
        while not rec.loaded:
            task = self._loading.get(name)
            if task is None:
                rec.status = "loading"
                task = asyncio.create_task(self._load(rec), name=f"lazyserve-load-{name}")
                task.add_done_callback(_consume_exception)
                self._loading[name] = task
            cold = True
            # shield: a caller that disconnects must not cancel the load for everyone else.
            await asyncio.shield(task)
        rec.in_use += 1
        rec.last_used = self._clock()
        return rec.model, cold

    def _release(self, rec: ModelRecord) -> None:
        rec.in_use -= 1
        rec.last_used = self._clock()

    async def _load(self, rec: ModelRecord) -> None:
        name = rec.name
        try:
            async with self._lock:
                if rec.loaded:
                    return
                estimate = rec.spec.memory_mb
                if estimate is None:
                    estimate = rec.measured_mb or 0.0
                self._make_room(rec, estimate)

                rss_before = self._rss()
                started = time.perf_counter()
                try:
                    loader = rec.spec.resolve_loader()
                    model = await asyncio.to_thread(loader, **rec.spec.options)
                    if not callable(getattr(model, "predict", None)):
                        raise TypeError(
                            f"loader returned {type(model).__name__}, which has no predict()"
                        )
                except Exception as exc:
                    rec.load_failures += 1
                    rec.last_error = f"{type(exc).__name__}: {exc}"
                    log.warning("load failed for %s: %s", name, rec.last_error)
                    raise LoadFailed(
                        name, f"loading model {name!r} failed: {rec.last_error}"
                    ) from exc
                elapsed = time.perf_counter() - started
                rss_after = self._rss()

                rec.model = model
                rec.status = "loaded"
                rec.measured_mb = max(0.0, (rss_after - rss_before) / MB)
                rec.load_seconds = elapsed
                rec.load_seconds_total += elapsed
                rec.loads += 1
                rec.last_error = None
                rec.last_used = self._clock()
                log.info(
                    "loaded %s in %.3fs (charged %.1f MB, %s; RSS delta %.1f MB)",
                    name,
                    elapsed,
                    rec.size_mb,
                    rec.size_source,
                    rec.measured_mb,
                )
                # A model without a declared size is only known after it loads.
                self._trim_to_budget(keep=rec)
        finally:
            if not rec.loaded:
                rec.status = "unloaded"
            self._loading.pop(name, None)

    def _make_room(self, incoming: ModelRecord, needed_mb: float) -> None:
        """Evict least recently used models until ``needed_mb`` fits in the budget."""
        budget = self.max_memory_mb
        if budget is None:
            return
        name = incoming.name
        others = [r for r in self._records.values() if r.loaded and r is not incoming]
        pinned_mb = sum(r.size_mb for r in others if r.spec.pin)
        if needed_mb > budget:
            raise ModelTooLarge(
                name,
                f"model {name!r} needs {needed_mb:.0f} MB but the whole budget is {budget:.0f} MB",
            )
        if needed_mb > budget - pinned_mb:
            raise ModelTooLarge(
                name,
                f"model {name!r} needs {needed_mb:.0f} MB but only {budget - pinned_mb:.0f} MB "
                f"of the {budget:.0f} MB budget is not held by pinned models",
            )
        used = sum(r.size_mb for r in others)
        if used + needed_mb <= budget:
            return
        for victim in self._eviction_order(others):
            self._unload_locked(victim, "lru")
            used -= victim.size_mb
            if used + needed_mb <= budget:
                return
        raise MemoryBusy(
            name,
            f"model {name!r} needs {needed_mb:.0f} MB; {used:.0f} of {budget:.0f} MB is held "
            "by models that are pinned or serving requests. Retry shortly.",
        )

    def _trim_to_budget(self, keep: ModelRecord) -> None:
        budget = self.max_memory_mb
        if budget is None:
            return
        used = self.memory_used_mb()
        if used <= budget:
            return
        others = [r for r in self._records.values() if r.loaded and r is not keep]
        for victim in self._eviction_order(others):
            self._unload_locked(victim, "lru")
            used -= victim.size_mb
            if used <= budget:
                return
        log.warning(
            "over budget after loading %s: %.1f of %.1f MB in use and nothing left to evict",
            keep.name,
            used,
            budget,
        )

    @staticmethod
    def _eviction_order(records: Iterable[ModelRecord]) -> list[ModelRecord]:
        candidates = [r for r in records if not r.spec.pin and r.in_use == 0]
        return sorted(candidates, key=lambda r: r.last_used or 0.0)

    def _unload_locked(self, rec: ModelRecord, reason: str) -> None:
        model = rec.model
        rec.model = None
        rec.status = "unloaded"
        rec.unloads[reason] += 1
        log.info("unloaded %s (%s), freed about %.1f MB", rec.name, reason, rec.size_mb)
        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.exception("close() failed for %s", rec.name)
        del model, close
        release_memory()

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.idle_check_interval_s)
            try:
                await self.sweep_idle()
            except Exception:  # pragma: no cover - defensive, keeps the sweeper alive
                log.exception("idle sweep failed")


def _consume_exception(task: asyncio.Task) -> None:
    # Waiters re-raise the error. This only stops "exception never retrieved"
    # warnings when every waiter went away before the load finished.
    if not task.cancelled():
        task.exception()


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)
