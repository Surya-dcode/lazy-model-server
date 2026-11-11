"""A model that holds a fixed amount of memory, for demonstrating eviction.

The block is an anonymous ``mmap`` filled with random bytes:

- Random bytes do not compress, so the operating system cannot shrink them
  in the background (macOS compresses idle pages, which hides constant data
  from RSS).
- ``close()`` unmaps the block, so the memory goes back to the operating
  system at once. Memory from the regular allocator is not always returned
  right away; see "Measuring memory" in the README.
"""

from __future__ import annotations

import mmap
import os
import time
from typing import Any

MB = 1024 * 1024
_CHUNK = 16 * MB


class BallastModel:
    def __init__(self, size_mb: float) -> None:
        self.size_mb = size_mb
        size = max(1, int(size_mb * MB))
        self._block: mmap.mmap | None = mmap.mmap(-1, size)
        written = 0
        while written < size:
            n = min(_CHUNK, size - written)
            self._block.write(os.urandom(n))
            written += n

    def predict(self, inputs: Any) -> dict[str, Any]:
        # Read one byte per page, the way a real model reads all of its weights
        # on every forward pass. Pages the OS swapped out come back into RSS.
        block = self._block
        if block is None:
            raise RuntimeError("model was closed")
        checksum = sum(block[i] for i in range(0, len(block), mmap.PAGESIZE)) & 0xFFFF
        return {"echo": inputs, "ballast_mb": self.size_mb, "checksum": checksum}

    def close(self) -> None:
        if self._block is not None:
            self._block.close()
            self._block = None


def load(size_mb: float = 100, load_delay_s: float = 0.0) -> BallastModel:
    """Allocate ``size_mb`` of memory. ``load_delay_s`` imitates reading weights from disk."""
    if load_delay_s:
        time.sleep(load_delay_s)
    return BallastModel(size_mb)
