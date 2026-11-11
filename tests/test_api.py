from __future__ import annotations

import pytest
from conftest import CountingLoader, spec
from fastapi.testclient import TestClient

from lazyserve.server import create_app


@pytest.fixture
def loaders():
    return {
        "clf": CountingLoader("clf"),
        "a": CountingLoader("a"),
        "b": CountingLoader("b"),
        "flaky": CountingLoader("flaky", fail_times=1),
    }


@pytest.fixture
def client(make_registry, loaders):
    specs = [
        spec("clf", loaders["clf"], memory_mb=10, pin=True),
        spec("a", loaders["a"], memory_mb=100),
        spec("b", loaders["b"], memory_mb=100),
        spec("flaky", loaders["flaky"], memory_mb=10),
        spec("huge", memory_mb=5000),
    ]
    registry = make_registry(specs, max_memory_mb=150, warm=["clf"])
    with TestClient(create_app(registry)) as c:
        yield c


def by_name(client) -> dict[str, dict]:
    return {m["name"]: m for m in client.get("/models").json()["models"]}


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 1  # the warm model
    assert body["models_configured"] == 5
    assert body["budget_mb"] == 150
    assert body["process_rss_mb"] > 0


def test_warm_list_loaded_at_startup(client, loaders):
    assert loaders["clf"].calls == 1
    models = by_name(client)
    assert models["clf"]["status"] == "loaded"
    assert models["a"]["status"] == "unloaded"

    resp = client.post("/predict/clf", json={"inputs": [1]})
    assert resp.json()["cold_start"] is False


def test_predict_cold_then_warm(client, loaders):
    first = client.post("/predict/a", json={"inputs": {"x": 1}})
    assert first.status_code == 200
    body = first.json()
    assert body["model"] == "a"
    assert body["outputs"] == {"tag": "a", "inputs": {"x": 1}}
    assert body["cold_start"] is True
    assert body["latency_ms"] >= 0

    second = client.post("/predict/a", json={"inputs": 2}).json()
    assert second["cold_start"] is False
    assert loaders["a"].calls == 1


def test_models_reports_status_size_and_hits(client):
    client.post("/predict/a", json={"inputs": 1})
    client.post("/predict/a", json={"inputs": 2})
    body = client.get("/models").json()
    assert body["budget_mb"] == 150
    assert body["used_mb"] == 110
    a = {m["name"]: m for m in body["models"]}["a"]
    assert a["status"] == "loaded"
    assert a["memory_mb"] == 100
    assert a["memory_source"] == "declared"
    assert a["hits"] == 2
    assert a["loads"] == 1
    assert a["load_seconds"] is not None
    assert a["last_used_s_ago"] is not None
    assert a["unloads"] == {"lru": 0, "idle": 0, "manual": 0}


def test_eviction_visible_through_api(client):
    client.post("/predict/a", json={"inputs": 1})
    client.post("/predict/b", json={"inputs": 1})  # 10 + 100 + 100 > 150: a goes
    models = by_name(client)
    assert models["a"]["status"] == "unloaded"
    assert models["a"]["unloads"]["lru"] == 1
    assert models["b"]["status"] == "loaded"
    assert models["clf"]["status"] == "loaded"


def test_load_and_unload_endpoints(client, loaders):
    resp = client.post("/models/a/load")
    assert resp.status_code == 200
    assert resp.json()["loaded_now"] is True
    assert resp.json()["status"] == "loaded"
    assert client.post("/models/a/load").json()["loaded_now"] is False
    assert loaders["a"].calls == 1

    resp = client.post("/models/a/unload")
    assert resp.json()["unloaded_now"] is True
    assert resp.json()["status"] == "unloaded"
    assert client.post("/models/a/unload").json()["unloaded_now"] is False


def test_unknown_model_is_404(client):
    for resp in (
        client.post("/predict/nope", json={"inputs": 1}),
        client.post("/models/nope/load"),
        client.post("/models/nope/unload"),
    ):
        assert resp.status_code == 404
        assert resp.json()["error"] == "unknown_model"


def test_model_bigger_than_budget_is_507(client):
    resp = client.post("/predict/huge", json={"inputs": 1})
    assert resp.status_code == 507
    body = resp.json()
    assert body["error"] == "model_too_large"
    assert body["model"] == "huge"
    assert "5000 MB" in body["detail"] and "150 MB" in body["detail"]


def test_load_failure_then_retry(client, loaders):
    first = client.post("/predict/flaky", json={"inputs": 1})
    assert first.status_code == 500
    assert first.json()["error"] == "load_failed"
    assert by_name(client)["flaky"]["last_error"].startswith("OSError")

    second = client.post("/predict/flaky", json={"inputs": 1})
    assert second.status_code == 200
    assert loaders["flaky"].calls == 2


def test_invalid_input_is_422(client):
    resp = client.post("/predict/a", json={"inputs": "bad"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_input"


def test_metrics_endpoint_content_type(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert 'lazyserve_model_loaded{model="clf"} 1' in resp.text


def test_example_config_serves_classifier():
    from pathlib import Path

    from lazyserve.server import create_app_from_config

    config = Path(__file__).resolve().parent.parent / "examples" / "models.yaml"
    with TestClient(create_app_from_config(config)) as c:
        body = c.post("/predict/classifier", json={"inputs": [[1.5, 1.2], [-1.0, -2.0]]}).json()
        assert body["outputs"]["labels"] == [1, 0]
        assert body["cold_start"] is False  # on the warm list
        assert c.post("/predict/classifier", json={"inputs": [[1]]}).status_code == 422
