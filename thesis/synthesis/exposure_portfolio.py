"""Exposure portfolio test: show the convergence signal is not single-variable.

Runs the headline partial correlation under five alternative exposures from the
same CESM-derived CSV:

1. delta_from_prev_T_field_meanabs (primary — field-mean |ΔT|)
2. delta_from_prev_T_global_abs (global mean |ΔT|)
3. delta_from_prev_P_global_abs (|ΔP| — precipitation)
4. (derived) max-cell |ΔT| approximated by field_meanabs / sign_agreement_frac
5. delta_from_prev_landfrac_field_meanabs (paleogeography null: land area change)

If climate-flavoured exposures give r ≈ 0.3–0.4 and paleogeography gives r ≈ 0,
the climate story is robust.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis._lib import build_controls, circular_shift_p, ensure_dir


EXPOSURES = [
    ("field_mean_dT", "delta_from_prev_T_field_meanabs"),
    ("global_dT", "delta_from_prev_T_global_abs"),
    ("global_dP", "delta_from_prev_P_global_abs"),
    ("land_area_change", "delta_from_prev_landfrac_field_meanabs"),
    ("coastline_change", "delta_from_prev_coastline_abs"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_exposure_portfolio")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    y = bins["functional_excess_similarity_js"].to_numpy(dtype=float)
    controls = build_controls(bins)

    results: dict[str, object] = {}

    for label, col in EXPOSURES:
        if col not in bins.columns:
            results[label] = {"error": f"column {col} not found in merged table"}
            continue
        v = bins[col].to_numpy(dtype=float)
        valid = np.isfinite(v) & np.isfinite(y)
        if valid.sum() < 8:
            results[label] = {"error": f"too few valid bins ({valid.sum()})"}
            continue
        test = circular_shift_p(v[valid], y[valid], controls[valid])
        results[label] = {"column": col, **test}

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    lines = [
        "# Exposure portfolio test",
        "",
        "Partial correlation of convergence with each exposure, controlling for",
        "time + sampling_PCA_PC12 + provinciality.",
        "",
        "| Exposure | Column | r | p (exact shift) | n |",
        "|----------|--------|---|-----------------|---|",
    ]
    for label, col in EXPOSURES:
        r = results.get(label, {})
        if "error" in r:
            lines.append(f"| {label} | {col} | — | {r['error']} | — |")
        else:
            lines.append(
                f"| {label} | {r.get('column', col)} | "
                f"{r.get('corr', float('nan')):.3f} | "
                f"{r.get('p_shift', float('nan')):.3g} | "
                f"{r.get('n', '—')} |"
            )

    climate_cols = ["field_mean_dT", "global_dT", "global_dP"]
    geo_cols = ["land_area_change", "coastline_change"]
    climate_rs = [abs(results[k].get("corr", 0)) for k in climate_cols if "corr" in results.get(k, {})]
    geo_rs = [abs(results[k].get("corr", 0)) for k in geo_cols if "corr" in results.get(k, {})]

    lines.extend([
        "",
        "## Summary",
        f"- Mean |r| for climate exposures: {np.mean(climate_rs):.3f}" if climate_rs else "- Climate exposures: insufficient data",
        f"- Mean |r| for paleogeography nulls: {np.mean(geo_rs):.3f}" if geo_rs else "- Paleogeography nulls: insufficient data",
    ])
    if climate_rs and geo_rs:
        if np.mean(climate_rs) > 0.2 and np.mean(geo_rs) < 0.15:
            lines.append("- Interpretation: climate story is robust; paleogeography does not drive the signal.")
        elif np.mean(geo_rs) >= 0.15:
            lines.append("- Interpretation: paleogeography shows non-trivial association; cannot cleanly separate from climate.")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
