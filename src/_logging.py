"""Structured JSONL logging for the paleontology-analytics pipeline."""

import json
import logging
import sys
from datetime import datetime, timezone


class _JSONFormatter(logging.Formatter):
    """Emits one JSON object per log record (JSONL)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields the caller attached via the `extra` dict.
        # Skip internal LogRecord attributes so only user-supplied keys appear.
        _BUILTIN = vars(logging.LogRecord("", 0, "", 0, "", (), None))
        for key, value in vars(record).items():
            if key not in _BUILTIN and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes JSONL to *stderr* at INFO level.

    Safe to call multiple times with the same *name*; handlers are only
    attached once.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
