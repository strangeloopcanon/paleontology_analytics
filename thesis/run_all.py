from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce key research outputs (marine volatility→convergence).")
    p.add_argument("--python", default=sys.executable, help="Python interpreter to use for all steps.")
    p.add_argument(
        "--skip-pair-model",
        action="store_true",
        help="Skip pair-level convergence model (uses existing output if present).",
    )
    p.add_argument(
        "--skip-role-jobs",
        action="store_true",
        help="Skip role/job interpretability step (uses existing output if present).",
    )
    p.add_argument(
        "--skip-low-energy",
        action="store_true",
        help="Skip low-energy index mediation step (uses existing output if present).",
    )
    p.add_argument(
        "--skip-time-series",
        action="store_true",
        help="Skip time-series/hierarchical inference step (uses existing output if present).",
    )
    args = p.parse_args()

    py = str(args.python)
    root = Path(__file__).resolve().parents[1]
    if not (root / "thesis").exists():
        raise SystemExit("Expected to run from the repo; could not find `thesis/`.")

    if not args.skip_pair_model:
        _run([py, "thesis/synthesis/pair_level_convergence_model.py"])

    if not args.skip_role_jobs:
        _run([py, "thesis/synthesis/role_job_drivers_volatility.py"])

    if not args.skip_low_energy:
        _run([py, "thesis/synthesis/low_energy_index_mediation.py"])

    if not args.skip_time_series:
        _run([py, "thesis/synthesis/time_series_hierarchical_models.py"])

    print("\nDone.")
    print("Key summaries to read:")
    print("- thesis/synthesis/FINAL_REPORT.md")
    print("- thesis/synthesis/output_pair_level_model_volatility_v1/summary.md")
    print("- thesis/synthesis/output_role_jobs_volatility_v1/summary.md")
    print("- thesis/synthesis/output_low_energy_index_mediation_v1/summary.md")
    print("- thesis/synthesis/output_time_series_hierarchical_models_v1/summary.md")


if __name__ == "__main__":
    main()
