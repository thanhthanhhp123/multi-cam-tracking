"""Logger dùng chung. Không dùng print() trong code chạy thật (CLAUDE.md §8)."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Cấu hình root logger. Gọi một lần ở entrypoint; gọi lại là no-op."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    from common.config import log_level  # import trễ để tránh vòng lặp import

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level or log_level())

    # Redis log rất ồn ở mức DEBUG.
    logging.getLogger("redis").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
