"""Reproduce the marine volatility→convergence analysis pipeline.

Run order reflects data dependencies:
  1. Core convergence pipeline (produces merged.csv used by everything else)
  2. Primary inference (time-series, pair-level, robustness)
  3. Sensitivity / hardening scripts
  4. Manuscript figures

Usage:
    python thesis/run_all.py
    python thesis/run_all.py --skip-core      # if core convergence output already exists
    python thesis/run_all.py --only-hardening  # run only sensitivity scripts
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, label: str = "") -> None:
    banner = label or " ".join(cmd)
    print(f"\n{'='*60}\n  {banner}\n{'='*60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce marine volatility→convergence results.")
    p.add_argument("--python", default=sys.executable, help="Python interpreter.")
    p.add_argument("--skip-core", action="store_true", help="Skip core convergence pipeline (use existing output).")
    p.add_argument("--only-hardening", action="store_true", help="Run only sensitivity/hardening scripts.")
    p.add_argument("--skip-figures", action="store_true", help="Skip figure generation.")
    args = p.parse_args()

    py = args.python
    root = Path(__file__).resolve().parents[1]
    if not (root / "thesis").exists():
        raise SystemExit("Expected to run from the repo root; could not find `thesis/`.")

    # ---- Phase 1: Core pipeline ----
    if not args.skip_core and not args.only_hardening:
        _run(
            [py, "thesis/convergence/run_convergence_analysis.py"],
            label="Core convergence analysis (produces timebin_metrics, pairwise_sample, ecospace mapping)",
        )
        _run(
            [py, "thesis/synthesis/robust_convergence_sampling_autocorr.py"],
            label="Sampling-controlled convergence + circular-shift tests (produces merged.csv)",
        )

    # ---- Phase 2: Primary inference ----
    if not args.only_hardening:
        _run(
            [py, "thesis/synthesis/pair_level_convergence_model.py"],
            label="Pair-level convergence model (cluster-robust SEs, mixed-effects)",
        )
        _run(
            [py, "thesis/synthesis/time_series_hierarchical_models.py"],
            label="Time-series hierarchical models (OLS, HAC, SARIMAX, MixedLM)",
        )

    # ---- Phase 3: Sensitivity / hardening ----
    _run(
        [py, "thesis/synthesis/ecospace_missingness_diagnostic.py"],
        label="Ecospace missingness diagnostic (coverage vs convergence)",
    )
    _run(
        [py, "thesis/synthesis/robustness_battery.py"],
        label="Robustness battery (LOO, block bootstrap, Lagerstaetten, SARIMAX sweep, HAC, coverage control)",
    )
    _run(
        [py, "thesis/synthesis/era_heterogeneity_investigation.py"],
        label="Era heterogeneity investigation (Mesozoic concentration)",
    )
    _run(
        [py, "thesis/synthesis/clade_restriction_test.py"],
        label="Clade restriction test (Bivalvia, Gastropoda, Brachiopoda)",
    )
    _run(
        [py, "thesis/synthesis/grid_sensitivity.py"],
        label="Grid sensitivity (10°, 15°, 20°)",
    )
    _run(
        [py, "thesis/synthesis/terrestrial_convergence_pilot.py"],
        label="Terrestrial convergence pilot",
    )

    # ---- Phase 4: Figures ----
    if not args.skip_figures:
        _run(
            [py, "thesis/manuscript_convergence_volatility/generate_figures.py"],
            label="Manuscript figures",
        )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Pipeline complete")
    print("=" * 60)
    print("\nKey outputs:")
    print("  Core:       thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/")
    print("  Robustness: thesis/synthesis/output_robustness_battery/")
    print("  Coverage:   thesis/synthesis/output_ecospace_missingness/")
    print("  Era:        thesis/synthesis/output_era_heterogeneity/")
    print("  Clades:     thesis/synthesis/output_clade_restriction/")
    print("  Grid:       thesis/synthesis/output_grid_sensitivity/")
    print("  Figures:    thesis/manuscript_convergence_volatility/figures/")
    print("  Report:     thesis/synthesis/FINAL_REPORT.md")


if __name__ == "__main__":
    main()
