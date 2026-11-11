"""Serve ML models that load on first use and unload to stay inside a memory budget."""

__version__ = "0.1.0"

from .config import ConfigError, ModelSpec, ServerConfig, load_config  # noqa: E402
from .registry import (  # noqa: E402
    InvalidInput,
    LoadFailed,
    MemoryBusy,
    ModelRegistry,
    ModelTooLarge,
    RegistryError,
    UnknownModel,
)

__all__ = [
    "ConfigError",
    "InvalidInput",
    "LoadFailed",
    "MemoryBusy",
    "ModelRegistry",
    "ModelSpec",
    "ModelTooLarge",
    "RegistryError",
    "ServerConfig",
    "UnknownModel",
    "__version__",
    "load_config",
]
