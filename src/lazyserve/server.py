"""HTTP API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import __version__
from .config import load_config
from .memory import MB
from .metrics import CONTENT_TYPE, render_metrics
from .registry import MemoryBusy, ModelRegistry, RegistryError


class PredictRequest(BaseModel):
    inputs: Any = None


def create_app(registry: ModelRegistry) -> FastAPI:
    """Build the FastAPI app around an existing registry."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await registry.start()
        try:
            yield
        finally:
            await registry.stop()

    app = FastAPI(title="lazy-model-server", version=__version__, lifespan=lifespan)
    app.state.registry = registry

    @app.exception_handler(RegistryError)
    async def registry_error(request: Request, exc: RegistryError) -> JSONResponse:
        headers = {"Retry-After": "1"} if isinstance(exc, MemoryBusy) else None
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "model": exc.model, "detail": exc.message},
            headers=headers,
        )

    @app.post("/predict/{model}")
    async def predict(model: str, body: PredictRequest) -> dict[str, Any]:
        started = time.perf_counter()
        result = await registry.predict(model, body.inputs)
        return {
            "model": model,
            "outputs": result.outputs,
            "cold_start": result.cold_start,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @app.get("/models")
    async def models() -> dict[str, Any]:
        return {
            "budget_mb": registry.max_memory_mb,
            "used_mb": round(registry.memory_used_mb(), 2),
            "models": registry.statuses(),
        }

    @app.post("/models/{name}/load")
    async def load(name: str) -> dict[str, Any]:
        cold = await registry.load(name)
        return {"loaded_now": cold, **registry.status(name)}

    @app.post("/models/{name}/unload")
    async def unload(name: str) -> dict[str, Any]:
        unloaded = await registry.unload(name)
        return {"unloaded_now": unloaded, **registry.status(name)}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "models_loaded": sum(1 for r in registry.records() if r.loaded),
            "models_configured": len(registry.records()),
            "used_mb": round(registry.memory_used_mb(), 2),
            "budget_mb": registry.max_memory_mb,
            "process_rss_mb": round(registry.rss_bytes() / MB, 1),
        }

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(render_metrics(registry), media_type=CONTENT_TYPE)

    return app


def create_app_from_config(path: str | Path) -> FastAPI:
    return create_app(ModelRegistry.from_config(load_config(path)))
