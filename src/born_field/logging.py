"""Structured logging setup.

Console-rendered and human-readable by default; JSON when ``BORNFIELD_LOG_JSON``
is set, which is what the container images use. Log events carry the fields a
risk service is actually asked about after the fact -- model version, cell id,
whether the request was refused -- so that "why did this cell score high on
Tuesday" is answerable from logs rather than by re-running the model.

**All log output goes to stderr.** structlog's default factory prints to stdout,
which would interleave diagnostics with the CLI's machine-readable output and
break ``bornfield config --json | jq``. stdout carries data, stderr carries
diagnostics.
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


class _LazyStderr:
    """Resolves ``sys.stderr`` at write time rather than at configure time.

    ``PrintLoggerFactory(file=sys.stderr)`` binds the stream object once, so any
    later reassignment of ``sys.stderr`` -- by a process supervisor, a
    daemonising wrapper, or pytest's capture -- leaves the logger writing to a
    handle that may be closed. Deferring the lookup costs one attribute access
    per record and removes that whole class of failure.
    """

    def write(self, message: str) -> int:
        """Write to whatever ``sys.stderr`` currently is."""
        return sys.stderr.write(message)

    def flush(self) -> None:
        """Flush whatever ``sys.stderr`` currently is."""
        sys.stderr.flush()


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog and the stdlib root logger. Idempotent."""
    global _configured  # process-wide setup, intentionally done once
    if _configured:
        return

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=_LazyStderr()),  # type: ignore[arg-type]
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level.upper())
    _configured = True


def get_logger(name: str, **initial: object) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, optionally pre-loaded with context."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger.bind(**initial) if initial else logger
