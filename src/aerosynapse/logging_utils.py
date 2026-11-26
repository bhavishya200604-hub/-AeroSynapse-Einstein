"""Structured logging helpers (JSON-lines friendly, no global mutable configuration)."""

from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``aerosynapse.<name>``) with a NullHandler attached."""
    logger = logging.getLogger(f"aerosynapse.{name}")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured event as a single JSON line at INFO level."""
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))
