from __future__ import annotations

import pytest

from lazyserve.config import ConfigError, ModelSpec, import_object, load_config, parse_config

GOOD = """
max_memory_mb: 512
idle_check_interval_s: 2
warm: [clf]
models:
  - name: clf
    loader: lazyserve.examples.logistic:load
    memory_mb: 5
    pin: true
  - name: ballast
    loader: lazyserve.examples.ballast.load
    idle_ttl_s: 30
    options: {size_mb: 10}
"""


def test_load_config(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(GOOD)
    config = load_config(path)
    assert config.max_memory_mb == 512
    assert config.idle_check_interval_s == 2
    assert config.warm == ["clf"]
    clf, ballast = config.models
    assert clf.pin is True and clf.memory_mb == 5
    assert ballast.memory_mb is None and ballast.idle_ttl_s == 30
    assert ballast.options == {"size_mb": 10}
    assert callable(ballast.resolve_loader())


def test_defaults():
    config = parse_config({"models": [{"name": "m", "loader": "pkg.mod:load"}]})
    assert config.max_memory_mb is None
    assert config.warm == []
    m = config.models[0]
    assert (m.pin, m.memory_mb, m.idle_ttl_s, m.options) == (False, None, None, {})


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "non-empty 'models'"),
        ({"models": [{"name": "m"}]}, "missing 'loader'"),
        ({"models": [{"name": "m", "loader": "x:y", "memroy_mb": 1}]}, "unknown keys memroy_mb"),
        ({"models": [{"name": "m", "loader": "x:y"}], "budget": 1}, "unknown keys budget"),
        ({"models": [{"name": "m", "loader": "x:y"}] * 2}, "duplicate model names: m"),
        ({"models": [{"name": "m", "loader": "x:y"}], "warm": ["n"]}, "unknown models: n"),
        ({"models": [{"name": "m", "loader": "not a path"}]}, "must look like"),
        ({"models": [{"name": "bad/name", "loader": "x:y"}]}, "model name"),
        ({"models": [{"name": "m", "loader": "x:y", "memory_mb": -1}]}, "positive"),
        ({"models": [{"name": "m", "loader": "x:y", "idle_ttl_s": 0}]}, "positive"),
        ({"models": [{"name": "m", "loader": "x:y", "pin": "yes"}]}, "pin"),
        ({"models": [{"name": "m", "loader": "x:y"}], "max_memory_mb": "1GB"}, "positive"),
    ],
)
def test_invalid_configs(data, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("models: [\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_import_object_forms():
    from lazyserve.examples import logistic

    assert import_object("lazyserve.examples.logistic:load") is logistic.load
    assert import_object("lazyserve.examples.logistic.load") is logistic.load
    with pytest.raises(ImportError, match="no attribute"):
        import_object("lazyserve.examples.logistic:missing")


def test_callable_loader_spec():
    def loader():
        return None

    assert ModelSpec(name="m", loader=loader).resolve_loader() is loader
