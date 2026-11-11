"""Process memory helpers."""

from __future__ import annotations

import gc
import sys

import psutil

MB = 1024 * 1024

_process = psutil.Process()


def process_rss_bytes() -> int:
    """Resident set size of this process, in bytes."""
    return _process.memory_info().rss


def release_memory() -> None:
    """Collect garbage and, if torch is already imported, empty the CUDA cache.

    torch is only touched when some model has already imported it, so a server
    that never serves a torch model never pays for importing it here.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - depends on the local CUDA setup
        pass
