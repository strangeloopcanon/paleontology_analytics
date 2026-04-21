"""Typed interface for the analysis module."""
from __future__ import annotations

from pydantic import BaseModel


class DashboardExportResult(BaseModel):
    """Outcome of a dashboard data export."""

    output_files: list[str]
    n_records: int
