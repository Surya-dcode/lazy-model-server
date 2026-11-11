"""Prometheus text exposition for the registry."""

from __future__ import annotations

from .registry import UNLOAD_REASONS, ModelRegistry

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def family(self, name: str, kind: str, help_text: str) -> None:
        self.lines.append(f"# HELP {name} {help_text}")
        self.lines.append(f"# TYPE {name} {kind}")

    def sample(self, name: str, value: float, **labels: str) -> None:
        if labels:
            body = ",".join(f'{k}="{escape_label(v)}"' for k, v in labels.items())
            self.lines.append(f"{name}{{{body}}} {_fmt(value)}")
        else:
            self.lines.append(f"{name} {_fmt(value)}")


def render_metrics(registry: ModelRegistry) -> str:
    records = registry.records()
    w = _Writer()

    w.family("lazyserve_model_loads_total", "counter", "Successful model loads.")
    for r in records:
        w.sample("lazyserve_model_loads_total", r.loads, model=r.name)

    w.family("lazyserve_model_load_failures_total", "counter", "Model loads that raised an error.")
    for r in records:
        w.sample("lazyserve_model_load_failures_total", r.load_failures, model=r.name)

    w.family(
        "lazyserve_model_unloads_total",
        "counter",
        "Model unloads by reason: lru (evicted for the memory budget), idle (TTL), manual.",
    )
    for r in records:
        for reason in UNLOAD_REASONS:
            w.sample(
                "lazyserve_model_unloads_total", r.unloads[reason], model=r.name, reason=reason
            )

    w.family("lazyserve_model_load_seconds", "summary", "Time spent in the model loader.")
    for r in records:
        w.sample("lazyserve_model_load_seconds_sum", round(r.load_seconds_total, 6), model=r.name)
        w.sample("lazyserve_model_load_seconds_count", r.loads, model=r.name)

    w.family("lazyserve_model_requests_total", "counter", "Predict requests per model.")
    for r in records:
        w.sample("lazyserve_model_requests_total", r.hits, model=r.name)

    w.family("lazyserve_model_loaded", "gauge", "1 if the model is in memory, else 0.")
    for r in records:
        w.sample("lazyserve_model_loaded", 1 if r.loaded else 0, model=r.name)

    w.family(
        "lazyserve_model_memory_mb",
        "gauge",
        "Memory charged to the model against the budget, 0 when unloaded.",
    )
    for r in records:
        w.sample("lazyserve_model_memory_mb", round(r.size_mb, 3) if r.loaded else 0, model=r.name)

    w.family("lazyserve_memory_used_mb", "gauge", "Memory charged to all loaded models.")
    w.sample("lazyserve_memory_used_mb", round(registry.memory_used_mb(), 3))

    if registry.max_memory_mb is not None:
        w.family("lazyserve_memory_budget_mb", "gauge", "Configured max_memory_mb.")
        w.sample("lazyserve_memory_budget_mb", registry.max_memory_mb)

    w.family("lazyserve_process_rss_bytes", "gauge", "Resident set size of the server process.")
    w.sample("lazyserve_process_rss_bytes", registry.rss_bytes())

    return "\n".join(w.lines) + "\n"
