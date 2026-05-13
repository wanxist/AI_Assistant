"""Unified logging configuration."""

import logging
import sys

from src.config import settings


def setup_logging() -> None:
    """Configure logging for the entire application.

    Called once at application startup (see api/main.py lifespan).
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on reload
    root.handlers = [handler]

    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "urllib3", "openai", "botocore"):
        logging.getLogger(name).setLevel(logging.WARNING)
