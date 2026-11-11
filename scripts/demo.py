"""Run the server locally and print a timeline of loads, evictions and process RSS.

Usage:
    python scripts/demo.py                # print the report
    python scripts/demo.py --out DEMO.md  # also write it to a file

The demo starts the server twice as a subprocess:

1. Eager: every model on the warm list and no budget, the way many services
   load models at startup. Only the startup RSS is recorded.
2. Lazy: one small pinned model on the warm list and a 240 MB budget. A fixed
   request pattern then exercises cold starts, LRU eviction and idle unload.

The "models" are ballast models that hold a fixed amount of memory, so the
numbers show the mechanism, not the speed of any real model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
import yaml

MB = 1024 * 1024

MODELS = [
    {
        "name": "classifier",
        "loader": "lazyserve.examples.logistic:load",
        "memory_mb": 5,
        "pin": True,
    },
    {
        "name": "embed",
        "loader": "lazyserve.examples.ballast:load",
        "memory_mb": 80,
        "idle_ttl_s": 60,
        "options": {"size_mb": 80, "load_delay_s": 0.4},
    },
    {
        "name": "rerank",
        "loader": "lazyserve.examples.ballast:load",
        # No memory_mb: the server charges the RSS delta it measures after loading.
        "idle_ttl_s": 60,
        "options": {"size_mb": 100, "load_delay_s": 0.6},
    },
    {
        "name": "summarize",
        "loader": "lazyserve.examples.ballast:load",
        "memory_mb": 120,
        "idle_ttl_s": 4,
        "options": {"size_mb": 120, "load_delay_s": 0.8},
    },
]

EAGER_CONFIG = {"warm": [m["name"] for m in MODELS], "models": MODELS}
LAZY_CONFIG = {
    "max_memory_mb": 240,
    "idle_check_interval_s": 1,
    "warm": ["classifier"],
    "models": MODELS,
}

PATTERN = [
    "classifier",
    "classifier",
    "embed",
    "embed",
    "rerank",
    "classifier",
    "summarize",
    "rerank",
    "embed",
    "summarize",
]


class Server:
    def __init__(self, config: dict, workdir: Path, label: str) -> None:
        self.config_path = workdir / f"{label}.yaml"
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> Server:
        cmd = [
            sys.executable,
            "-m",
            "lazyserve",
            "--config",
            str(self.config_path),
            "--port",
            str(self.port),
            "--log-level",
            "warning",
        ]
        self.proc = subprocess.Popen(cmd)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                self.get("/healthz")
                return self
            except (urllib.error.URLError, ConnectionError):
                if self.proc.poll() is not None:
                    raise RuntimeError("server exited during startup") from None
                time.sleep(0.1)
        raise RuntimeError("server did not become healthy in 60 s")

    def __exit__(self, *exc) -> None:
        assert self.proc is not None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def rss_mb(self) -> float:
        assert self.proc is not None
        return psutil.Process(self.proc.pid).memory_info().rss / MB

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=30) as resp:
            return json.loads(resp.read())

    def post(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    def models(self) -> dict[str, dict]:
        return {m["name"]: m for m in self.get("/models")["models"]}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _unload_events(before: dict[str, dict], after: dict[str, dict]) -> list[str]:
    events = []
    for name, model in after.items():
        for reason, count in model["unloads"].items():
            if count > before[name]["unloads"][reason]:
                verb = "evicted" if reason == "lru" else f"unloaded ({reason})"
                events.append(f"{verb} {name}")
    return events


def _loaded(models: dict[str, dict]) -> str:
    return ", ".join(n for n, m in models.items() if m["status"] == "loaded") or "-"


def run_demo() -> str:
    out: list[str] = []
    emit = out.append

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        with Server(EAGER_CONFIG, workdir, "eager") as eager:
            # One request per model, so every model has been used, as in a live service.
            for m in MODELS:
                eager.post(f"/predict/{m['name']}", {"inputs": [[0.5, -0.2]]})
            eager_rss = eager.rss_mb()
            eager_used = eager.get("/healthz")["used_mb"]

        with Server(LAZY_CONFIG, workdir, "lazy") as lazy:
            time.sleep(0.2)
            t0 = time.monotonic()
            rows: list[tuple] = []
            state = {"before": lazy.models()}

            def record(step, start, latency, events):
                after = lazy.models()
                events = _unload_events(state["before"], after) + events
                charged = sum(
                    m["memory_mb"] or 0 for m in after.values() if m["status"] == "loaded"
                )
                row = (
                    f"{time.monotonic() - t0:.1f}",
                    step,
                    start,
                    latency,
                    _loaded(after),
                    charged,
                    lazy.rss_mb(),
                    "; ".join(events) or "-",
                )
                rows.append(row)
                state["before"] = after
                return after

            def predict(step, name):
                body = lazy.post(f"/predict/{name}", {"inputs": [[0.5, -0.2]]})
                start = "cold" if body["cold_start"] else "warm"
                latency = f"{body['latency_ms']:.0f}"
                after = record(step, start, latency, [])
                if body["cold_start"]:
                    load = f"load {name} ({after[name]['load_seconds']:.2f} s)"
                    events = rows[-1][7]
                    rows[-1] = rows[-1][:7] + (load if events == "-" else f"{load}; {events}",)

            record("startup", "-", "-", ["warm load classifier"])
            start_rss = rows[0][6]
            for i, name in enumerate(PATTERN, start=1):
                predict(f"#{i} predict {name}", name)
            peak_rss = max(row[6] for row in rows)

            # Stay idle long enough for summarize's 4 s TTL to pass.
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                time.sleep(0.5)
                if lazy.models()["summarize"]["status"] != "loaded":
                    break
            record("idle", "-", "-", [])
            # One more request so embed's pages are resident when RSS is read.
            predict(f"#{len(PATTERN) + 1} predict embed", "embed")
            final_rss = rows[-1][6]
            final = lazy.models()
            metrics_text = (
                urllib.request.urlopen(lazy.base + "/metrics", timeout=10).read().decode()
            )

    emit("# Demo output\n")
    emit(
        "This is a local demo run of `scripts/demo.py`, not a benchmark. The models are "
        "ballast models that hold a fixed block of memory, so the numbers show how lazy "
        "loading and eviction behave, not how fast any real model is. Your numbers will differ. "
        "RSS comes from the operating system: if the machine is short on memory it can swap "
        "pages out, and RSS then reads lower than the memory the models hold.\n"
    )
    emit(
        f"- Date: {dt.date.today().isoformat()}\n"
        f"- Machine: {platform.system()} {platform.machine()}, Python {platform.python_version()}\n"
        f"- Budget for the lazy run: {LAZY_CONFIG['max_memory_mb']} MB, "
        "idle check every 1 s\n"
    )
    emit("## Models\n")
    emit("| model | holds | declared size | pinned | idle TTL |")
    emit("|---|---|---|---|---|")
    for m in MODELS:
        holds = f"{m['options']['size_mb']} MB" if "options" in m else "tiny (logistic regression)"
        emit(
            f"| {m['name']} | {holds} | {str(m.get('memory_mb', 'none, measured'))} "
            f"| {'yes' if m.get('pin') else 'no'} | {m.get('idle_ttl_s', '-')} |"
        )
    emit("")
    emit("## Timeline (lazy run)\n")
    emit(
        "`charged MB` is what the server counts against the budget. "
        "`RSS MB` is the server process's resident memory, read with psutil after each step.\n"
    )
    emit("| t (s) | step | start | latency ms | loaded after | charged MB | RSS MB | events |")
    emit("|---|---|---|---|---|---|---|---|")
    for t, step, start, latency, loaded, charged, rss, events in rows:
        emit(
            f"| {t} | {step} | {start} | {latency} | {loaded} | {charged:.0f} | {rss:.0f} "
            f"| {events} |"
        )
    emit("")
    loads = sum(m["loads"] for m in final.values())
    evictions = sum(m["unloads"]["lru"] for m in final.values())
    idle = sum(m["unloads"]["idle"] for m in final.values())
    emit("## Summary\n")
    emit("| | RSS MB |")
    emit("|---|---|")
    emit(f"| Eager: all models loaded at startup (charged {eager_used:.0f} MB) | {eager_rss:.0f} |")
    emit(f"| Lazy: at startup, only the pinned classifier loaded | {start_rss:.0f} |")
    emit(f"| Lazy: peak during the request pattern | {peak_rss:.0f} |")
    emit(f"| Lazy: at the end, after the idle unload | {final_rss:.0f} |")
    emit("")
    emit(
        f"The lazy run made {len(PATTERN) + 1} requests, did {loads} loads, "
        f"{evictions} LRU evictions and {idle} idle unload(s).\n"
    )
    emit("## /metrics at the end of the lazy run (excerpt)\n")
    emit("```")
    keep = (
        "lazyserve_model_loads_total",
        "lazyserve_model_unloads_total",
        "lazyserve_memory_",
        "lazyserve_model_load_seconds_count",
    )
    for line in metrics_text.splitlines():
        if line.startswith(keep) and not line.endswith(" 0"):
            emit(line)
    emit("```")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="also write the report to this file")
    args = parser.parse_args()
    report = run_demo()
    print(report)
    if args.out:
        Path(args.out).write_text(report)


if __name__ == "__main__":
    main()
