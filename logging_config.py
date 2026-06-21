"""
logging_config.py — Production-grade structured logging.

Replaces the original bare `logging.basicConfig(level=logging.INFO)` with:

  * a rotating file handler (so logs don't grow unbounded on a long-running
    research server),
  * a console handler for local development,
  * a structured (key=value) log line format that's easy to grep and easy to
    parse into a log aggregator later without a rewrite,
  * a `log_timing` decorator/context-manager for recording execution timings
    (model loads, preprocessing passes, per-job durations) without scattering
    `time.time()` calls through business logic,
  * a cache hit/miss counter helper used by the embedding cache and model
    manager to report effectiveness.

Usage
-----
    from logging_config import get_logger, log_timing

    logger = get_logger(__name__)
    logger.info("Job started", extra={"job_id": job_id})

    with log_timing(logger, "model_load", model_name=model_name):
        model = AutoModel.from_pretrained(...)
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

_CONFIGURED = False


class _KeyValueFormatter(logging.Formatter):
    """
    Formats log records as: timestamp level logger message key=value key=value

    Falls back gracefully when no extra fields are supplied, so this is a
    drop-in replacement for the default formatter rather than a breaking
    change to log shape.
    """

    _RESERVED = set(logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "
            f"{record.levelname:<8} "
            f"{record.name} — {record.getMessage()}"
        )
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self._RESERVED and not k.startswith("_")
        }
        if extras:
            kv = " ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
            base = f"{base} | {kv}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure the root logger once for the whole process.

    Safe to call multiple times — subsequent calls are no-ops, matching the
    idempotent behaviour expected of a module that may be imported from
    several entry points (the FastAPI app, a CLI script, a test runner).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "app.log"

    root = logging.getLogger()
    root.setLevel(level)

    formatter = _KeyValueFormatter()

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the logging system on first use."""
    configure_logging()
    return logging.getLogger(name)


@contextmanager
def log_timing(logger: logging.Logger, operation: str, **context) -> Iterator[None]:
    """
    Context manager that logs the duration of *operation* on exit, including
    whether it succeeded or raised. Extra keyword args are attached as
    structured fields (e.g. model_name="xlm-roberta-base").

    Example
    -------
        with log_timing(logger, "model_load", model_name="xlm-roberta-base"):
            model = AutoModel.from_pretrained(...)
    """
    start = time.perf_counter()
    logger.info(f"{operation} started", extra={"operation": operation, **context})
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            f"{operation} failed",
            extra={"operation": operation, "elapsed_seconds": round(elapsed, 4), **context},
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(
            f"{operation} completed",
            extra={"operation": operation, "elapsed_seconds": round(elapsed, 4), **context},
        )


class CacheHitMissCounter:
    """
    Lightweight in-process hit/miss counter for reporting cache effectiveness
    in logs (e.g. embedding cache, model manager singleton reuse).

    Not persisted across restarts — this is for live operational visibility,
    not analytics; long-term cache effectiveness should be read from the
    embedding cache's own row counts.
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger(f"cache.{name}")
        self.hits = 0
        self.misses = 0

    def hit(self) -> None:
        self.hits += 1

    def miss(self) -> None:
        self.misses += 1

    @property
    def hit_rate(self) -> Optional[float]:
        total = self.hits + self.misses
        return (self.hits / total) if total else None

    def report(self) -> None:
        self.logger.info(
            f"{self.name} cache stats",
            extra={"hits": self.hits, "misses": self.misses,
                   "hit_rate": self.hit_rate},
        )
