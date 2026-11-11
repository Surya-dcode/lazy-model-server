"""Command line entry point: ``lazyserve --config models.yaml``."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .config import ConfigError
from .server import create_app_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lazyserve",
        description="Serve models that load on first use and unload to fit a memory budget.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("LAZYSERVE_CONFIG", "models.yaml"),
        help="YAML config file (default: $LAZYSERVE_CONFIG or models.yaml)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        app = create_app_from_config(args.config)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0
