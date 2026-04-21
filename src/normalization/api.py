"""Typed interface for the normalization module."""
from __future__ import annotations

from pydantic import BaseModel


class NormalizeResult(BaseModel):
    """Outcome of a normalization pass."""

    source: str
    n_rows: int
    output_path: str
    columns: list[str]
