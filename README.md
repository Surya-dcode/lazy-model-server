# lazy-model-server

A small FastAPI model server that loads each model the first time it is asked for, and unloads
models to stay inside a memory budget.

[![CI](https://github.com/Surya-dcode/lazy-model-server/actions/workflows/ci.yml/badge.svg)](https://github.com/Surya-dcode/lazy-model-server/actions/workflows/ci.yml)

## The problem

A common way to serve several models from one container is to load all of them at startup.
That is simple, but the container then pays for the memory of every model all the time, even
when most requests go to one or two of them. The container has to be sized for the sum of all
models, and adding a model makes every replica bigger.

## How this fixes it

- **Lazy loading.** A model is loaded on its first request, not at startup. A model that is
  never called never uses memory.
- **Single-flight loads.** If 20 requests for a cold model arrive at once, the loader runs
  once and the other 19 requests wait for that same load.
- **Memory budget with LRU eviction.** Set `max_memory_mb`. Before a model loads, if the
  loaded models plus the new one would go over the budget, the server unloads the least
  recently used models until the new one fits.
- **Idle unload.** A model can have an `idle_ttl_s`. A background task unloads it once it has
  gone that long without a request.
- **Unload frees memory.** The server drops its reference to the model, calls the model's
  `close()` if it has one, runs `gc.collect()`, and calls `torch.cuda.empty_cache()` if torch
  is already imported.

The container can then be sized for the budget instead of for the sum of all models.

## The trade-off: cold starts

The first request to an unloaded model waits for the load. In the demo below that is 0.7 to
1.25 seconds against 0 to 3 ms for a warm request, and for a large real model it can be much
longer. Two settings control which models pay this cost:

- **`pin: true`** keeps a model loaded once it is loaded. It is never evicted and never idle
  unloaded. Use it for the models that serve most of the traffic.
- **`warm: [...]`** loads models at startup, so even their first request is warm. A model that
  is warm but not pinned can still be evicted later.

A good starting point is to pin and warm the one or two hot models and let the rest load on
demand. The `cold_start` field in every predict response, and the load metrics, show how often
requests are paying for a load.

Other limits to know about:

- **Loads are serialized.** Two different models never load at the same time. This keeps
  the RSS measurement of a load attributable to one model and keeps the budget arithmetic
  simple, at the cost of a queue when several cold models are requested together.
- **A model in use is not evicted.** If the only models that could be evicted are serving
  requests, a load returns `503` with `Retry-After: 1` instead of waiting.
- **One budget per process.** With several uvicorn workers, each worker has its own registry
  and its own budget. Set `max_memory_mb` per worker.
- **GPU memory is not measured.** Declare `memory_mb` for GPU models. See below.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
lazyserve --config examples/models.yaml
```

In another shell:

```bash
curl -s localhost:8000/predict/classifier \
  -H 'content-type: application/json' \
  -d '{"inputs": [[1.5, 1.2], [-1.0, -2.0]]}'
# {"model":"classifier","outputs":{"labels":[1,0],"probabilities":[...]},"cold_start":false,...}

curl -s localhost:8000/predict/ballast-small -H 'content-type: application/json' -d '{"inputs": 1}'
curl -s localhost:8000/models
curl -s localhost:8000/metrics
```

The `mlp` model in `examples/models.yaml` needs the torch extra: `pip install -e ".[torch]"`.
torch is imported inside its loader, so the server starts and serves the other models without
torch installed, and only a request to `mlp` imports it. Without the extra, a request to `mlp`
returns `500 load_failed` naming the missing module.

## Writing a model

A loader is any function that returns an object with a `predict(inputs)` method. `inputs` is
whatever JSON the client sent under `"inputs"`, and the return value must be JSON
serializable.

```python
class MyModel:
    def __init__(self, path):
        self.weights = read_weights(path)

    def predict(self, inputs):
        return {"scores": score(self.weights, inputs)}

    def close(self):  # optional, called on unload
        self.weights = None


def load(path="weights.bin"):
    return MyModel(path)
```

```yaml
models:
  - name: my-model
    loader: mypackage.models:load
    memory_mb: 900
    options: {path: /models/my-model.bin}
```

If `predict` raises `ValueError` or `TypeError`, the server answers `422 invalid_input`. Any
other exception is `500 predict_failed`.

## Configuration reference

Top level:

| key | type | default | meaning |
|---|---|---|---|
| `models` | list | required | the models the server can serve |
| `max_memory_mb` | number | none | memory budget for all loaded models; omit for no budget |
| `idle_check_interval_s` | number | `10` | how often the idle sweeper runs |
| `warm` | list of names | `[]` | models to load at startup, in order |

Per model:

| key | type | default | meaning |
|---|---|---|---|
| `name` | string | required | used in URLs; letters, digits, `.`, `_`, `-` |
| `loader` | string | required | `package.module:function` or `package.module.function` |
| `memory_mb` | number | none | declared size; if missing, the measured RSS delta of the load is used |
| `pin` | bool | `false` | never evict or idle unload this model |
| `idle_ttl_s` | number | none | unload after this many seconds without a request |
| `options` | mapping | `{}` | keyword arguments passed to the loader |

Unknown keys are rejected, so a typo such as `memroy_mb` fails at startup instead of being
ignored. A warm model that fails to load is logged and the server still starts; the next
request for it tries again.

The `lazyserve` command takes `--config` (default `$LAZYSERVE_CONFIG`, then `models.yaml`),
`--host`, `--port` and `--log-level`. `python -m lazyserve` does the same.

## Endpoints

| method and path | what it does |
|---|---|
| `POST /predict/{model}` | body `{"inputs": ...}`; loads the model if needed; returns `outputs`, `cold_start`, `latency_ms` |
| `GET /models` | every model with status, size and its source, load time, seconds since last use, hits, loads, unloads by reason, last error |
| `POST /models/{name}/load` | load now (for example from a deploy hook) |
| `POST /models/{name}/unload` | unload now |
| `GET /healthz` | liveness, number of loaded models, charged memory, process RSS |
| `GET /metrics` | Prometheus text format |

Errors are JSON: `{"error": "<code>", "model": "<name>", "detail": "<message>"}`.

| status | code | when |
|---|---|---|
| 404 | `unknown_model` | no model with that name |
| 422 | `invalid_input` | `predict` raised `ValueError` or `TypeError` |
| 500 | `load_failed` | the loader raised; the model stays unloaded and the next request retries |
| 500 | `predict_failed` | `predict` raised anything else |
| 503 | `memory_busy` | the model would fit, but the models that could be evicted are serving requests |
| 507 | `model_too_large` | the model is bigger than the budget, or than what pinned models leave free |

## Metrics

| metric | type | labels |
|---|---|---|
| `lazyserve_model_loads_total` | counter | `model` |
| `lazyserve_model_load_failures_total` | counter | `model` |
| `lazyserve_model_unloads_total` | counter | `model`, `reason` (`lru` = budget eviction, `idle`, `manual`) |
| `lazyserve_model_load_seconds` | summary (`_sum`, `_count`) | `model` |
| `lazyserve_model_requests_total` | counter | `model` |
| `lazyserve_model_loaded` | gauge | `model` |
| `lazyserve_model_memory_mb` | gauge | `model` |
| `lazyserve_memory_used_mb` | gauge | |
| `lazyserve_memory_budget_mb` | gauge | (only when a budget is set) |
| `lazyserve_process_rss_bytes` | gauge | |

A high rate of `lazyserve_model_unloads_total{reason="lru"}` followed by loads of the same
model means the budget is too small for the traffic: models are being evicted and reloaded.

## Measuring memory

Each model is charged against the budget by its declared `memory_mb` if there is one, and
otherwise by how much the process RSS (resident set size, read with psutil) grew during its
load. The measured delta is always recorded and shown in `/models` as `measured_mb`, so the
two can be compared.

The measured delta is a rough number. Things that make it wrong:

- **Allocator caching.** Python and the C allocator keep freed memory for reuse. A load can
  reuse memory freed by an earlier unload and show a delta near zero, and an unload may not
  lower RSS at all because the allocator keeps the pages. glibc returns large blocks to the
  OS when they are freed; small objects often stay. The ballast example uses `mmap` so its
  memory is returned at once and the demo shows the drop.
- **First import.** The first model that imports a library such as torch is charged for the
  library as well as its weights.
- **Lazy pages.** Memory-mapped weight files are only counted once their pages are touched,
  which can be after the load returns.
- **Swapping and compression.** If the machine is short on memory, the OS can swap pages out
  or (on macOS) compress them, and RSS then reads lower than what the model holds. While
  building the demo, an earlier ballast filled with a constant byte showed a small fraction
  of its size in RSS a second after loading, most likely because macOS compressed the idle
  pages. The current ballast uses random bytes and reads its pages on every request.
- **Other threads.** Anything else that allocates during a load is charged to that model.
  Loads are serialized to keep this small, but requests to already loaded models still run.
- **GPU memory.** RSS does not include CUDA memory at all.

For anything in production, declare `memory_mb` from a measurement you trust (for example a
peak measured in a separate process) and use the measured value as a check.

A model that has no declared size is only known after it loads. If that load pushes the total
over the budget, the server evicts other models to get back under it. The next time that model
loads, its last measured size is used for the check before loading.

## Demo

`scripts/demo.py` starts the server as a subprocess twice, runs a fixed request pattern, and
prints a timeline. The full output of one local run is in [DEMO.md](DEMO.md). The numbers
below are from that run on a laptop (macOS, arm64, Python 3.11), not a benchmark. The models
are ballast models that hold a fixed block of random bytes (80, 100 and 120 MB) plus a tiny
pinned logistic regression classifier, with a 240 MB budget.

| t (s) | step | start | latency ms | loaded after | charged MB | RSS MB | events |
|---|---|---|---|---|---|---|---|
| 0.0 | startup | - | - | classifier | 5 | 51 | warm load classifier |
| 0.7 | #3 predict embed | cold | 676 | classifier, embed | 85 | 147 | load embed (0.67 s) |
| 0.7 | #4 predict embed | warm | 1 | classifier, embed | 85 | 147 | - |
| 1.7 | #5 predict rerank | cold | 960 | classifier, embed, rerank | 189 | 251 | load rerank (0.96 s) |
| 2.9 | #7 predict summarize | cold | 1249 | classifier, rerank, summarize | 229 | 290 | load summarize (1.23 s); evicted embed |
| 3.6 | #9 predict embed | cold | 696 | classifier, embed, rerank | 189 | 249 | load embed (0.68 s); evicted summarize |
| 4.8 | #10 predict summarize | cold | 1211 | classifier, embed, summarize | 205 | 263 | load summarize (1.20 s); evicted rerank |
| 9.9 | idle | - | - | classifier, embed | 85 | 143 | unloaded (idle) summarize |

(Some warm steps are left out here; DEMO.md has all of them.)

| | RSS MB |
|---|---|
| Eager: all models loaded at startup | 371 |
| Lazy: at startup | 51 |
| Lazy: peak during the pattern | 290 |
| Lazy: at the end, after the idle unload | 143 |

What the run shows:

- With every model loaded at startup the process held 371 MB. With lazy loading it started at
  51 MB and never went above 290 MB, even though all four models were used.
- `rerank` has no declared size. Its measured RSS delta was 104 MB for a 100 MB block, which
  is the kind of overhead described above.
- Each eviction removed the least recently used model, and the pinned classifier was never
  evicted although it was often the least recently used.
- The process RSS is 45 to 60 MB above the charged memory because it includes the Python
  interpreter, FastAPI and uvicorn. The budget covers models only.

Run it yourself with `python scripts/demo.py` (add `--out DEMO.md` to save it). Results vary
between runs and machines, more so on a machine that is short on memory.

## Running tests

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

The tests make no network calls and download nothing. They cover:

- 20 concurrent first requests cause exactly one load
- LRU eviction order, and evicting no more models than needed
- pinned models are never evicted or idle unloaded
- a model bigger than the budget, or than what pinned models leave, gets a clear 507
- a busy model is not evicted, and the caller gets a 503
- idle unload with a hand-driven clock, and the background sweeper
- a loader exception leaves the model unloaded and the next request retries, including when
  several requests were waiting on the failed load
- size from the measured RSS delta, and the post-load trim back under budget
- unload drops the last reference to the model
- every endpoint through FastAPI's TestClient
- the `/metrics` output parses as Prometheus text format

## Layout

```
src/lazyserve/
  config.py      YAML config and validation
  registry.py    lazy loading, single-flight, budget, LRU, idle unload
  server.py      FastAPI app
  metrics.py     Prometheus text output
  memory.py      RSS reading and memory release
  cli.py         the lazyserve command
  examples/      logistic regression, ballast and torch example models
examples/models.yaml
scripts/demo.py
tests/
```

## License

MIT. See [LICENSE](LICENSE).
