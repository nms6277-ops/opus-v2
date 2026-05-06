"""Lightweight structured logging setup."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", logs_dir: Path | None = None) -> None:
    """Configure root logger once. Idempotent."""
    root = logging.getLogger()
    if getattr(root, "_opus_configured", False):
        return

    root.setLevel(level.upper())
    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            logs_dir / "opus.log",
            maxBytes=50 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence chatty third-party loggers
    for name in (
        "asyncio",
        "websockets.client",
        "websockets.server",
        "urllib3",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    root._opus_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
