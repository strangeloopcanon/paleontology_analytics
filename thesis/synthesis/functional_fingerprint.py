"""Functional fingerprint: which ecological roles expand under volatile climates?

For each time bin, computes role-frequency residuals vs the global mean role
distribution. Splits bins into high-volatility and low-volatility groups
(median split) and identifies which roles consistently over- or under-represent
in volatile intervals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from matplotlib import pyplot as plt
from thesis._lib import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairwise",
        default="thesis/convergence/output/pairwise_sample.csv",
    )
    ap.add_argument(
        "--ecospace",
        default="thesis/convergence/output/ecospace_genus_mapping.csv",
    )
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument(
        "--pbdb",
        default="data/processed/merged_occurrences.parquet",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_functional_fingerprint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    eco = pd.read_csv(args.ecospace)
    eco = eco.dropna(subset=["genus", "role_id"]).copy()
    eco = eco[eco["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()
    genus_role = dict(zip(eco["genus"], eco["role_id"]))

    # Read occurrence data to get genus × time_bin.
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
    occ = occ.dropna(subset=["role_id"]).copy()

    # Role frequency per bin (fraction of genera in each role).
    role_counts = occ.groupby(["time_bin", "role_id"])["genus"].nunique().reset_index(name="n_genera")
    bin_totals = role_counts.groupby("time_bin")["n_genera"].sum().rename("total")
    role_counts = role_counts.merge(bin_totals, on="time_bin")
    role_counts["frac"] = role_counts["n_genera"] / role_counts["total"]

    # Pivot to matrix: bins × roles.
    role_matrix = role_counts.pivot(index="time_bin", columns="role_id", values="frac").fillna(0)
    role_matrix = role_matrix.loc[role_matrix.index.isin(bins["time_bin"])].copy()
    role_matrix = role_matrix.sort_index(ascending=False)

    if role_matrix.empty:
        print("No role data after filtering.")
        return

    global_mean = role_matrix.mean(axis=0)
    residuals = role_matrix.subtract(global_mean, axis=1)

    vol_series = bins.set_index("time_bin")["delta_from_prev_T_field_meanabs"]
    common_bins = sorted(set(residuals.index) & set(vol_series.index), reverse=True)
    residuals = residuals.loc[common_bins]
    vol = vol_series.loc[common_bins]

    vol_median = float(vol.median())
    high_vol = vol >= vol_median
    low_vol = vol < vol_median

    high_mean = residuals.loc[high_vol].mean(axis=0)
    low_mean = residuals.loc[low_vol].mean(axis=0)
    diff = high_mean - low_mean
    diff = diff.sort_values(ascending=False)

    results = {
        "n_bins": len(common_bins),
        "n_high_vol": int(high_vol.sum()),
        "n_low_vol": int(low_vol.sum()),
        "vol_median_threshold": vol_median,
        "n_roles": len(diff),
        "top_expanding_roles": {
            k: {"diff": float(diff[k]), "high_vol_mean_frac": float(high_mean[k]), "low_vol_mean_frac": float(low_mean[k])}
            for k in diff.head(10).index
        },
        "top_contracting_roles": {
            k: {"diff": float(diff[k]), "high_vol_mean_frac": float(high_mean[k]), "low_vol_mean_frac": float(low_mean[k])}
            for k in diff.tail(10).index
        },
    }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Figure: top/bottom roles.
    top_n = min(15, len(diff))
    top = diff.head(top_n)
    bottom = diff.tail(top_n)
    combined = pd.concat([top, bottom])

    fig, ax = plt.subplots(figsize=(8, max(6, len(combined) * 0.3)))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in combined.values]
    ax.barh(range(len(combined)), combined.values, color=colors, alpha=0.8)
    labels = [r.replace("|", " | ") for r in combined.index]
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δ fractional representation (high-vol − low-vol)")
    ax.set_title("Functional fingerprint of volatile climates")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_fingerprint.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Functional fingerprint",
        "",
        f"- {results['n_bins']} bins, {results['n_high_vol']} high-volatility, {results['n_low_vol']} low-volatility",
        f"- {results['n_roles']} distinct ecological roles",
        "",
        "## Roles that EXPAND under volatile climates",
    ]
    for k, v in results["top_expanding_roles"].items():
        lines.append(f"- {k.replace('|', ' | ')}: Δ = {v['diff']:+.4f} (high={v['high_vol_mean_frac']:.4f}, low={v['low_vol_mean_frac']:.4f})")
    lines.extend(["", "## Roles that CONTRACT under volatile climates"])
    for k, v in results["top_contracting_roles"].items():
        lines.append(f"- {k.replace('|', ' | ')}: Δ = {v['diff']:+.4f} (high={v['high_vol_mean_frac']:.4f}, low={v['low_vol_mean_frac']:.4f})")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
