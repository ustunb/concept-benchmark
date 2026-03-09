"""Logging utilities for the concept-benchmark package.

Default format matches the previous print() output so that users see
identical messages at INFO level.
"""
from __future__ import annotations

import logging
import sys

__all__ = ["get_logger", "setup_logging"]

_DEFAULT_FORMAT = "%(message)s"
_VERBOSE_FORMAT = "%(levelname)s %(name)s: %(message)s"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped under ``concept_benchmark``."""
    return logging.getLogger(f"concept_benchmark.{name}")


def setup_logging(level: int = logging.INFO, verbose_format: bool = False) -> None:
    """Configure the root ``concept_benchmark`` logger.

    Called once at CLI entry.  Idempotent — second calls are no-ops unless
    the handler list is empty.

    Args:
        level: Logging level (e.g. logging.DEBUG, logging.INFO, logging.WARNING).
        verbose_format: If True, include level name and logger name in output.
    """
    global _configured
    if _configured:
        return
    root = logging.getLogger("concept_benchmark")
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = _VERBOSE_FORMAT if verbose_format else _DEFAULT_FORMAT
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    _configured = True
