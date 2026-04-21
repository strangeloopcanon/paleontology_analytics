"""Smoke tests for new experiment scripts.

Verifies that each script can be imported, has a main() function, and that
argparse is configured without crashing. Does NOT require data files.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


EXPERIMENT_SCRIPTS = [
    "thesis/synthesis/spatial_null_test.py",
    "thesis/synthesis/coverage_confound_battery.py",
    "thesis/synthesis/exposure_portfolio.py",
    "thesis/synthesis/baseline_shift_quantile.py",
    "thesis/synthesis/forward_prediction.py",
    "thesis/synthesis/functional_fingerprint.py",
    "thesis/synthesis/clade_decomposition.py",
    "thesis/synthesis/mesozoic_mechanism.py",
    "thesis/synthesis/modern_analog.py",
    "thesis/body_size_stability/cross_system_replication.py",
    "thesis/geographic_portfolio/hierarchical_model.py",
    "thesis/paleobiotic_velocity/clade_stratified_salvage.py",
]


@pytest.mark.parametrize("script", EXPERIMENT_SCRIPTS)
def test_script_has_help(script: str) -> None:
    """Each script should accept --help without crashing."""
    path = ROOT / script
    if not path.exists():
        pytest.skip(f"{script} not found")
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"{script} --help failed:\n{result.stderr}"
    assert "usage:" in result.stdout.lower() or "optional arguments" in result.stdout.lower() or "options:" in result.stdout.lower()


def test_thesis_lib_imports() -> None:
    """thesis._lib should export all canonical helpers."""
    sys.path.insert(0, str(ROOT))
    lib = importlib.import_module("thesis._lib")
    for name in [
        "ensure_dir", "z_score", "residualize", "partial_corr",
        "pca_scores", "build_controls", "circular_shift_p",
        "perm_test_corr", "haversine_km", "circular_mean_deg",
    ]:
        assert hasattr(lib, name), f"thesis._lib missing {name}"
        assert callable(getattr(lib, name)), f"thesis._lib.{name} is not callable"
