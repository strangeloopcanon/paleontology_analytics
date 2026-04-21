"""Test specific mechanisms for the Mesozoic concentration of the convergence signal.

Candidates:
(a) Role saturation: realized roles / theoretical max as a proxy for ecospace
    maturity. Low saturation = more "slack" for convergent filling.
(b) Post-Permian vacancy: fraction of roles present in the preceding bin that
    disappear (turnover proxy for ecological vacancy).
(c) Marine diversity trajectory: log genus richness per bin as a proxy for
    ecosystem complexity.

Tests mediation: does including each candidate as a control attenuate the
Mesozoic-specific volatility-convergence correlation?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis._lib import build_controls, ensure_dir, partial_corr


# Mesozoic boundaries (Ma).
MESO_START, MESO_END = 252.0, 66.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument(
        "--ecospace",
        default="thesis/convergence/output/ecospace_genus_mapping.csv",
    )
    ap.add_argument(
        "--pbdb",
        default="data/processed/merged_occurrences.parquet",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_mesozoic_mechanism")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    eco = pd.read_csv(args.ecospace)
    eco = eco.dropna(subset=["genus", "role_id"]).copy()
    eco = eco[eco["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()
    genus_role = dict(zip(eco["genus"], eco["role_id"]))

    try:
        occ = pd.read_parquet(args.pbdb, columns=["source_db", "mid_ma", "genus"])
        occ = occ[occ["source_db"] == "PBDB"].copy()
    except FileNotFoundError:
        print(f"Occurrence file not found: {args.pbdb}")
        return

    occ["genus"] = occ["genus"].astype(str).str.strip()
    occ = occ.dropna(subset=["genus", "mid_ma"]).copy()
    occ["time_bin"] = (pd.to_numeric(occ["mid_ma"], errors="coerce") / 10.0).round() * 10.0
    occ["role_id"] = occ["genus"].map(genus_role)

    # Compute per-bin mediator candidates.
    all_roles = sorted(eco["role_id"].unique())
    n_theoretical = len(all_roles)

    bin_roles = occ.dropna(subset=["role_id"]).groupby("time_bin")["role_id"].nunique().rename("n_realized_roles")
    bin_genera = occ.groupby("time_bin")["genus"].nunique().rename("n_marine_genera")
    mediators = pd.DataFrame({"time_bin": bins["time_bin"]})
    mediators = mediators.merge(bin_roles, on="time_bin", how="left")
    mediators = mediators.merge(bin_genera, on="time_bin", how="left")
    mediators["role_saturation"] = mediators["n_realized_roles"] / n_theoretical
    mediators["log_genus_richness"] = np.log1p(mediators["n_marine_genera"].to_numpy(dtype=float))

    # Role turnover: fraction of roles in bin t-1 that are absent in bin t.
    sorted_bins = sorted(mediators["time_bin"].unique(), reverse=True)
    bin_role_sets = occ.dropna(subset=["role_id"]).groupby("time_bin")["role_id"].apply(set).to_dict()
    turnover = {}
    for i in range(1, len(sorted_bins)):
        prev_b, curr_b = sorted_bins[i - 1], sorted_bins[i]
        prev_roles = bin_role_sets.get(prev_b, set())
        curr_roles = bin_role_sets.get(curr_b, set())
        if prev_roles:
            turnover[curr_b] = len(prev_roles - curr_roles) / len(prev_roles)
    mediators["role_turnover"] = mediators["time_bin"].map(turnover)

    bins = bins.merge(mediators[["time_bin", "role_saturation", "log_genus_richness", "role_turnover"]], on="time_bin", how="left")

    # Mesozoic subset.
    meso = bins[(bins["time_bin"] <= MESO_START) & (bins["time_bin"] >= MESO_END)].copy()

    y_meso = meso["functional_excess_similarity_js"].to_numpy(dtype=float)
    v_meso = meso["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    controls_meso = build_controls(meso)

    results: dict[str, object] = {
        "n_mesozoic_bins": len(meso),
        "baseline_mesozoic_r": partial_corr(v_meso, y_meso, controls_meso),
    }

    candidate_cols = ["role_saturation", "log_genus_richness", "role_turnover"]
    for col in candidate_cols:
        vals = meso[col].to_numpy(dtype=float)
        if np.isfinite(vals).sum() < 6:
            results[f"mediation_{col}"] = {"error": "insufficient data"}
            continue
        controls_ext = np.column_stack([controls_meso, vals])
        r_ext = partial_corr(v_meso, y_meso, controls_ext)
        attenuation = results["baseline_mesozoic_r"] - r_ext if np.isfinite(r_ext) else float("nan")
        results[f"mediation_{col}"] = {
            "partial_r_with_mediator": r_ext,
            "attenuation": attenuation,
            "pct_attenuation": 100 * attenuation / abs(results["baseline_mesozoic_r"]) if abs(results["baseline_mesozoic_r"]) > 1e-6 else float("nan"),
        }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    lines = [
        "# Mesozoic mechanism mediation",
        "",
        f"- Mesozoic bins: {results['n_mesozoic_bins']}",
        f"- Baseline Mesozoic partial r: {results['baseline_mesozoic_r']:.4f}",
        "",
    ]
    for col in candidate_cols:
        r = results.get(f"mediation_{col}", {})
        if "error" in r:
            lines.append(f"- {col}: {r['error']}")
        else:
            lines.append(
                f"- {col}: partial r with mediator = {r['partial_r_with_mediator']:.4f}, "
                f"attenuation = {r['attenuation']:.4f} ({r['pct_attenuation']:.1f}%)"
            )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
