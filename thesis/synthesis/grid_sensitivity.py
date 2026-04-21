"""Grid-size sensitivity: rerun convergence at 10, 15 and 20 degree grids.

Uses the same clade_restriction_test infrastructure but on the full occurrence set.
Uses the FULL primary control set (time + sampling_PCA_PC12 + provinciality) to
match the headline specification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis._lib import build_controls, ensure_dir

from clade_restriction_test import compute_convergence_for_subset, _clean, _partial_corr_shift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    ap.add_argument("--ecospace", default="thesis/convergence/output/ecospace_genus_mapping.csv")
    ap.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    ap.add_argument("--out", default="thesis/synthesis/output_grid_sensitivity")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    occ = pd.read_parquet(
        args.pbdb,
        columns=["source_db", "occurrence_id", "mid_ma", "lat", "lng", "paleolat", "paleolng",
                  "phylum", "class", "order", "family", "genus"],
    )
    occ = occ[occ["source_db"] == "PBDB"].copy()
    occ["genus"] = occ["genus"].map(_clean)
    occ = occ.dropna(subset=["genus", "mid_ma"]).copy()
    occ["analysis_lat"] = occ["paleolat"].where(occ["paleolat"].notna(), occ["lat"])
    occ["analysis_lng"] = occ["paleolng"].where(occ["paleolng"].notna(), occ["lng"])
    occ = occ.dropna(subset=["analysis_lat", "analysis_lng"]).copy()

    eco = pd.read_csv(args.ecospace)
    eco["genus"] = eco["genus"].map(_clean)
    eco = eco.dropna(subset=["genus"]).copy()
    eco = eco[eco["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()
    eco = eco.dropna(subset=["role_id"]).copy()

    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})

    grids = [10, 15, 20]
    results = {}

    for grid_deg in grids:
        metrics = compute_convergence_for_subset(
            occ, eco,
            time_bin_myr=10.0, grid_deg=float(grid_deg),
            min_genera_per_region=max(10, 25 - (grid_deg - 10)),  # scale threshold with grid size
            max_pairs=30000, seed=args.seed,
        )
        merged = metrics.merge(earth, on="time_bin", how="left")
        merged = merged.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
        merged = merged.sort_values("time_bin", ascending=False).reset_index(drop=True)
        merged.to_csv(out_dir / f"metrics_grid{grid_deg}.csv", index=False)

        if len(merged) < 8:
            results[f"grid_{grid_deg}"] = {"error": f"too few bins ({len(merged)})"}
            continue

        y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
        v = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)

        # Full controls matching the primary specification.
        full_controls = build_controls(merged)
        test_full = _partial_corr_shift(v, y, full_controls)
        # Also keep time-only for comparison with prior runs.
        t = merged["time_bin"].to_numpy(dtype=float)
        test_time_only = _partial_corr_shift(v, y, t.reshape(-1, 1))
        results[f"grid_{grid_deg}"] = {
            "n_bins": len(merged),
            "test_full_controls": test_full,
            "test_time_only": test_time_only,
        }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Figure: grid comparison scatter.
    fig, axes = plt.subplots(1, len(grids), figsize=(5 * len(grids), 4.5), squeeze=False)
    for i, grid_deg in enumerate(grids):
        csv_path = out_dir / f"metrics_grid{grid_deg}.csv"
        if not csv_path.exists():
            continue
        m = pd.read_csv(csv_path)
        ax = axes[0][i]
        ax.scatter(m["delta_from_prev_T_field_meanabs"], m["functional_excess_similarity_js"],
                   s=40, alpha=0.8, color="#1f77b4")
        r = results.get(f"grid_{grid_deg}", {}).get("test_full_controls", {})
        ax.set_title(f"{grid_deg}° grid (full controls)\nr={r.get('corr', 'nan'):.3f}, p={r.get('p_shift', 'nan'):.3f}")
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Excess similarity")
    fig.tight_layout()
    fig.savefig(fig_dir / "grid_sensitivity.png", dpi=220)
    plt.close(fig)

    lines = ["# Grid-size sensitivity", "", "## Full primary controls (time + sampling_PCA_PC12 + provinciality)", ""]
    for grid_deg in grids:
        r = results.get(f"grid_{grid_deg}", {})
        if "error" in r:
            lines.append(f"- {grid_deg}° grid: {r['error']}")
        else:
            tf = r.get("test_full_controls", {})
            tt = r.get("test_time_only", {})
            lines.append(
                f"- {grid_deg}° grid (n={r['n_bins']}): "
                f"full-controls r={tf.get('corr', 'nan'):.3f}, p={tf.get('p_shift', 'nan'):.3g} | "
                f"time-only r={tt.get('corr', 'nan'):.3f}, p={tt.get('p_shift', 'nan'):.3g}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
