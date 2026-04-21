"""Typed interface for the acquisition module."""
from __future__ import annotations

from pydantic import BaseModel


class PBDBFetchRequest(BaseModel):
    """Parameters for a PBDB occurrence download."""

    interval: str
    columns: list[str]
    output_path: str


class PBDBFetchResult(BaseModel):
    """Outcome of a PBDB fetch operation."""

    interval: str
    columns: list[str]
    output_path: str
    n_rows_fetched: int
