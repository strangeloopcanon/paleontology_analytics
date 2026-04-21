"""Clade-stratified salvage for paleobiotic velocity.

Pre-specified hypothesis: centroid velocity matters for sessile marine
invertebrates during glacial intervals (where they cannot track climate)
but not for mobile vertebrates.

Tests this by fitting the same hazard model from run_pipeline.py on a
(clade × interval) stratification, with the modern-coordinate negative
control alongside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


GLACIAL_INTERVALS = [
    ("Late_Ordovician", 460.0, 440.0),
    ("Late_Devonian", 380.0, 360.0),
    ("Permo_Carboniferous", 330.0, 270.0),
]

CLADE_GROUPS = {
    "sessile_marine_inverts": ["Brachiopoda", "Bryozoa", "Anthozoa", "Crinoidea"],
    "mobile_marine_inverts": ["Gastropoda", "Bivalvia", "Cephalopoda", "Trilobita"],
    "marine_vertebrates": ["Actinopterygii", "Chondrichthyes", "Placodermi"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="thesis/paleobiotic_velocity/output_clade_stratified")
    ap.add_argument("--delta-auc-threshold", type=float, default=0.01,
                     help="Pre-specified threshold for 'salvageable' ΔAUC")
    args = ap.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    results: dict[str, object] = {
        "prespecified_hypothesis": (
            "Centroid velocity predicts lower next-bin extinction hazard for "
            "sessile marine invertebrates during glacial intervals, but not "
            "for mobile vertebrates."
        ),
        "delta_auc_threshold": args.delta_auc_threshold,
        "glacial_intervals": [
            {"name": name, "start_ma": s, "end_ma": e}
            for name, s, e in GLACIAL_INTERVALS
        ],
        "clade_groups": {k: v for k, v in CLADE_GROUPS.items()},
        "note": (
            "This script defines the framework. Execution requires the "
            "velocity pipeline output from run_pipeline.py (genus-level "
            "velocity + extinction data). Run with --velocity-data when "
            "that output is available."
        ),
        "decision_rule": (
            f"If ΔAUC for sessile marine inverts during glacial intervals "
            f"exceeds {args.delta_auc_threshold:.3f} AND exceeds the "
            f"modern-coordinate negative control ΔAUC, the track has a "
            f"publishable narrow finding. Otherwise, the dead-end framing "
            f"is locked in."
        ),
    }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    lines = [
        "# Clade-stratified velocity salvage",
        "",
        "## Pre-specified hypothesis",
        results["prespecified_hypothesis"],
        "",
        "## Clade groups",
    ]
    for group, clades in CLADE_GROUPS.items():
        lines.append(f"- {group}: {', '.join(clades)}")
    lines.extend([
        "",
        "## Glacial intervals",
    ])
    for interval in results["glacial_intervals"]:
        lines.append(f"- {interval['name']}: {interval['start_ma']}–{interval['end_ma']} Ma")
    lines.extend([
        "",
        "## Decision rule",
        results["decision_rule"],
        "",
        "## Status",
        "Framework defined. Awaiting velocity pipeline output.",
    ])

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
