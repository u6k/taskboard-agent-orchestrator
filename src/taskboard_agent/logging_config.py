from __future__ import annotations

import contextvars
import logging
import logging.config
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "taskboard_agent_trace_id",
    default="-",
)
_original_factory = logging.getLogRecordFactory()
_factory_installed = False


def configure_logging(config_file: str | Path = "logging.conf") -> None:
    install_log_record_factory()
    logging.config.fileConfig(config_file, disable_existing_loggers=False)


def install_log_record_factory() -> None:
    global _factory_installed
    if _factory_installed:
        return

    def record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = _original_factory(*args, **kwargs)
        if not hasattr(record, "trace_id"):
            record.trace_id = _trace_id.get()
        return record

    logging.setLogRecordFactory(record_factory)
    _factory_installed = True


@contextmanager
def log_trace(trace_id: str) -> Iterator[None]:
    install_log_record_factory()
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        _trace_id.reset(token)
