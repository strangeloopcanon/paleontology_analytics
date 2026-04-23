from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.analysis.advanced_sota import calculate_null_model
from src.analysis.export_web_data import export_dashboard_data
from src.analysis.kids import generate_kids_data
from src.cli import main as cli_main


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_cli_merge_uses_input_directory(tmp_path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    _write_parquet(
        input_dir / "pbdb_occurrences.parquet",
        [{"occurrence_id": "1", "genus": "A", "mid_ma": 100.0}],
    )

    argv_prev = sys.argv[:]
    try:
        sys.argv = [
            "cli.py",
            "normalize",
            "--source",
            "merge",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
        ]
        cli_main()
    finally:
        sys.argv = argv_prev

    assert (output_dir / "merged_occurrences.parquet").exists()


def test_generate_kids_data_handles_non_mesozoic_input(tmp_path) -> None:
    data_path = tmp_path / "data.parquet"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    rows = [
        {"genus": "A", "mid_ma": 10.0, "lat": 1.0, "lng": 2.0},
        {"genus": "B", "mid_ma": 20.0, "lat": 1.5, "lng": 2.5},
        {"genus": "C", "mid_ma": 30.0, "lat": 2.0, "lng": 3.0},
    ]
    _write_parquet(data_path, rows)

    generate_kids_data(str(data_path), str(output_dir))

    deep_time_path = output_dir / "deep_time_data.json"
    dino_path = output_dir / "dino_zone_data.json"
    assert deep_time_path.exists()
    assert dino_path.exists()

    dino_payload = json.loads(dino_path.read_text())
    assert "facts" in dino_payload
    assert isinstance(dino_payload["facts"], list)


def test_null_model_handles_none_modularity_output(tmp_path) -> None:
    data_path = tmp_path / "data.parquet"
    output_file = tmp_path / "null_model.json"

    rows = [
        {"genus": f"G{i}", "mid_ma": 100.0, "lat": float(i * 10), "lng": float(i * 10)}
        for i in range(12)
    ]
    _write_parquet(data_path, rows)

    payload = calculate_null_model(str(data_path), str(output_file), n_iterations=5)

    assert output_file.exists()
    assert payload is not None
    assert "observed_modularity" in payload


def test_export_dashboard_data_tolerates_missing_reference_and_empty_sota(tmp_path) -> None:
    data_path = tmp_path / "data.parquet"
    output_file = tmp_path / "web_data.json"
    explorer_file = tmp_path / "explorer_data.json"

    rows = [
        {"genus": f"G{i}", "mid_ma": 100.0, "lat": 1.0, "lng": 2.0}
        for i in range(50)
    ]
    _write_parquet(data_path, rows)

    export_dashboard_data(
        data_path=str(data_path),
        output_file=str(output_file),
        explorer_output_file=str(explorer_file),
    )

    assert output_file.exists()
    assert explorer_file.exists()

    summary = json.loads(output_file.read_text())
    assert "sota" in summary
    assert "time" in summary["sota"]
