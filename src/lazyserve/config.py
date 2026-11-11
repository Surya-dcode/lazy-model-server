"""Server and model configuration, read from YAML."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LOADER_RE = re.compile(r"^[A-Za-z_][\w.]*(:[A-Za-z_]\w*|\.[A-Za-z_]\w*)$")

_MODEL_KEYS = {"name", "loader", "memory_mb", "pin", "idle_ttl_s", "options"}
_SERVER_KEYS = {"max_memory_mb", "idle_check_interval_s", "warm", "models"}


class ConfigError(ValueError):
    """The configuration file is missing a field or has an invalid value."""


@dataclass
class ModelSpec:
    """One model the server can serve.

    ``loader`` is either an import path such as ``"mypkg.models:load"`` or a
    callable. It is called with ``options`` as keyword arguments and must return
    an object with a ``predict(inputs)`` method.
    """

    name: str
    loader: str | Callable[..., Any]
    memory_mb: float | None = None
    pin: bool = False
    idle_ttl_s: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise ConfigError(
                f"model name {self.name!r} must start with a letter or digit and use only "
                "letters, digits, '.', '_' or '-'"
            )
        if isinstance(self.loader, str):
            if not _LOADER_RE.match(self.loader):
                raise ConfigError(
                    f"model {self.name!r}: loader {self.loader!r} must look like "
                    "'package.module:function' or 'package.module.function'"
                )
        elif not callable(self.loader):
            raise ConfigError(f"model {self.name!r}: loader must be an import path or callable")
        if self.memory_mb is not None:
            self.memory_mb = _positive(self.memory_mb, f"model {self.name!r}: memory_mb")
        if self.idle_ttl_s is not None:
            self.idle_ttl_s = _positive(self.idle_ttl_s, f"model {self.name!r}: idle_ttl_s")
        if not isinstance(self.pin, bool):
            raise ConfigError(f"model {self.name!r}: pin must be true or false")
        if not isinstance(self.options, dict):
            raise ConfigError(f"model {self.name!r}: options must be a mapping")

    def resolve_loader(self) -> Callable[..., Any]:
        """Return the loader function, importing it if it is given as a path."""
        if callable(self.loader):
            return self.loader
        return import_object(self.loader)


@dataclass
class ServerConfig:
    """Settings for the whole server."""

    models: list[ModelSpec]
    max_memory_mb: float | None = None
    idle_check_interval_s: float = 10.0
    warm: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_memory_mb is not None:
            self.max_memory_mb = _positive(self.max_memory_mb, "max_memory_mb")
        self.idle_check_interval_s = _positive(self.idle_check_interval_s, "idle_check_interval_s")
        names = [m.name for m in self.models]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ConfigError(f"duplicate model names: {', '.join(dupes)}")
        unknown = [w for w in self.warm if w not in names]
        if unknown:
            raise ConfigError(f"warm list names unknown models: {', '.join(unknown)}")


def import_object(path: str) -> Any:
    """Import ``"pkg.mod:attr"`` or ``"pkg.mod.attr"`` and return the attribute."""
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ImportError(f"module {module_name!r} has no attribute {attr!r}") from None


def parse_config(data: Any) -> ServerConfig:
    """Build a :class:`ServerConfig` from already-parsed YAML data."""
    if not isinstance(data, dict):
        raise ConfigError("config must be a mapping with a 'models' list")
    _reject_unknown(data, _SERVER_KEYS, "server config")
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("config needs a non-empty 'models' list")
    models = []
    for i, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise ConfigError(f"models[{i}] must be a mapping")
        _reject_unknown(raw, _MODEL_KEYS, f"models[{i}]")
        for key in ("name", "loader"):
            if key not in raw:
                raise ConfigError(f"models[{i}] is missing '{key}'")
        if not isinstance(raw["loader"], str):
            raise ConfigError(f"models[{i}]: loader must be a string import path")
        models.append(
            ModelSpec(
                name=raw["name"],
                loader=raw["loader"],
                memory_mb=raw.get("memory_mb"),
                pin=raw.get("pin", False),
                idle_ttl_s=raw.get("idle_ttl_s"),
                options=raw.get("options") or {},
            )
        )
    warm = data.get("warm") or []
    if not isinstance(warm, list) or not all(isinstance(w, str) for w in warm):
        raise ConfigError("warm must be a list of model names")
    kwargs: dict[str, Any] = {"models": models, "warm": warm}
    if "max_memory_mb" in data:
        kwargs["max_memory_mb"] = data["max_memory_mb"]
    if "idle_check_interval_s" in data:
        kwargs["idle_check_interval_s"] = data["idle_check_interval_s"]
    return ServerConfig(**kwargs)


def load_config(path: str | Path) -> ServerConfig:
    """Read and validate a YAML config file."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return parse_config(data)


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"{label} must be a positive number, got {value!r}")
    return float(value)


def _reject_unknown(raw: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ConfigError(f"{where}: unknown keys {', '.join(extra)}")
