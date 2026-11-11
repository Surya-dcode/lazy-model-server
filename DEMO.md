# Demo output

This is a local demo run of `scripts/demo.py`, not a benchmark. The models are ballast models that hold a fixed block of memory, so the numbers show how lazy loading and eviction behave, not how fast any real model is. Your numbers will differ. RSS comes from the operating system: if the machine is short on memory it can swap pages out, and RSS then reads lower than the memory the models hold.

- Date: 2026-11-11
- Machine: Darwin arm64, Python 3.11.13
- Budget for the lazy run: 240 MB, idle check every 1 s

## Models

| model | holds | declared size | pinned | idle TTL |
|---|---|---|---|---|
| classifier | tiny (logistic regression) | 5 | yes | - |
| embed | 80 MB | 80 | no | 60 |
| rerank | 100 MB | none, measured | no | 60 |
| summarize | 120 MB | 120 | no | 4 |

## Timeline (lazy run)

`charged MB` is what the server counts against the budget. `RSS MB` is the server process's resident memory, read with psutil after each step.

| t (s) | step | start | latency ms | loaded after | charged MB | RSS MB | events |
|---|---|---|---|---|---|---|---|
| 0.0 | startup | - | - | classifier | 5 | 51 | warm load classifier |
| 0.0 | #1 predict classifier | warm | 0 | classifier | 5 | 51 | - |
| 0.0 | #2 predict classifier | warm | 0 | classifier | 5 | 51 | - |
| 0.7 | #3 predict embed | cold | 676 | classifier, embed | 85 | 147 | load embed (0.67 s) |
| 0.7 | #4 predict embed | warm | 1 | classifier, embed | 85 | 147 | - |
| 1.7 | #5 predict rerank | cold | 960 | classifier, embed, rerank | 189 | 251 | load rerank (0.96 s) |
| 1.7 | #6 predict classifier | warm | 0 | classifier, embed, rerank | 189 | 251 | - |
| 2.9 | #7 predict summarize | cold | 1249 | classifier, rerank, summarize | 229 | 290 | load summarize (1.23 s); evicted embed |
| 2.9 | #8 predict rerank | warm | 2 | classifier, rerank, summarize | 229 | 290 | - |
| 3.6 | #9 predict embed | cold | 696 | classifier, embed, rerank | 189 | 249 | load embed (0.68 s); evicted summarize |
| 4.8 | #10 predict summarize | cold | 1211 | classifier, embed, summarize | 205 | 263 | load summarize (1.20 s); evicted rerank |
| 9.9 | idle | - | - | classifier, embed | 85 | 143 | unloaded (idle) summarize |
| 9.9 | #11 predict embed | warm | 3 | classifier, embed | 85 | 143 | - |

## Summary

| | RSS MB |
|---|---|
| Eager: all models loaded at startup (charged 309 MB) | 371 |
| Lazy: at startup, only the pinned classifier loaded | 51 |
| Lazy: peak during the request pattern | 290 |
| Lazy: at the end, after the idle unload | 143 |

The lazy run made 11 requests, did 6 loads, 3 LRU evictions and 1 idle unload(s).

## /metrics at the end of the lazy run (excerpt)

```
lazyserve_model_loads_total{model="classifier"} 1
lazyserve_model_loads_total{model="embed"} 2
lazyserve_model_loads_total{model="rerank"} 1
lazyserve_model_loads_total{model="summarize"} 2
lazyserve_model_unloads_total{model="embed",reason="lru"} 1
lazyserve_model_unloads_total{model="rerank",reason="lru"} 1
lazyserve_model_unloads_total{model="summarize",reason="lru"} 1
lazyserve_model_unloads_total{model="summarize",reason="idle"} 1
lazyserve_model_load_seconds_count{model="classifier"} 1
lazyserve_model_load_seconds_count{model="embed"} 2
lazyserve_model_load_seconds_count{model="rerank"} 1
lazyserve_model_load_seconds_count{model="summarize"} 2
lazyserve_memory_used_mb 85
lazyserve_memory_budget_mb 240
```
