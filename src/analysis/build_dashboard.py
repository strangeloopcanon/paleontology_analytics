"""
Build all static dashboard assets under `dashboard/`.

This powers `dashboard/index.html` (static) and keeps JSONs in sync with the analysis code.
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.advanced_sota import calculate_climate_correlation, calculate_null_model, calculate_rates
from src.analysis.export_web_data import export_dashboard_data
from src.analysis.kids import generate_kids_data
from src.analysis.taxonomy import generate_taxonomy_data


def build_dashboard_assets(
    *,
    data_path: str = "data/processed/merged_occurrences.parquet",
    dashboard_dir: str = "dashboard",
) -> None:
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    export_dashboard_data(data_path=data_path, output_file=f"{dashboard_dir}/web_data.json")
    calculate_rates(data_path=data_path, output_file=f"{dashboard_dir}/rates_data.json")
    calculate_climate_correlation(data_path=data_path, output_file=f"{dashboard_dir}/climate_data.json")
    calculate_null_model(data_path=data_path, output_file=f"{dashboard_dir}/null_model_data.json", n_iterations=100)
    generate_taxonomy_data(data_path=data_path, output_dir=dashboard_dir)
    generate_kids_data(data_path=data_path, output_dir=dashboard_dir)


if __name__ == "__main__":
    build_dashboard_assets()

