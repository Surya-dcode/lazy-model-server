from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from lazyserve import ModelRegistry, ModelSpec


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class EchoModel:
    def __init__(self, tag: str, closed: list[str] | None = None) -> None:
        self.tag = tag
        self._closed = closed

    def predict(self, inputs: Any) -> dict[str, Any]:
        if inputs == "bad":
            raise ValueError("bad input")
        if inputs == "boom":
            raise RuntimeError("model crashed")
        return {"tag": self.tag, "inputs": inputs}

    def close(self) -> None:
        if self._closed is not None:
            self._closed.append(self.tag)


class CountingLoader:
    """A loader that counts calls and can be slow or fail."""

    def __init__(self, tag: str, delay_s: float = 0.0, fail_times: int = 0, closed=None) -> None:
        self.tag = tag
        self.delay_s = delay_s
        self.fail_times = fail_times
        self.closed = closed
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, **options: Any) -> EchoModel:
        with self._lock:
            self.calls += 1
            call = self.calls
        if self.delay_s:
            time.sleep(self.delay_s)
        if call <= self.fail_times:
            raise OSError(f"weights for {self.tag} not reachable (attempt {call})")
        return EchoModel(self.tag, self.closed)


def spec(name: str, loader=None, **kwargs: Any) -> ModelSpec:
    return ModelSpec(name=name, loader=loader or CountingLoader(name), **kwargs)


def constant_rss() -> int:
    return 100 * 1024 * 1024


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_registry(clock):
    def _make(specs, **kwargs: Any) -> ModelRegistry:
        kwargs.setdefault("clock", clock)
        kwargs.setdefault("rss", constant_rss)
        return ModelRegistry(specs, **kwargs)

    return _make
