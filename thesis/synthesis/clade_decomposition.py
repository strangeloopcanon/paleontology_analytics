"""Clade decomposition: which clade pairs drive convergence in volatile bins?

Instead of restricting to a single clade (which kills the signal), decompose
the pairwise JS distance into clade-pair contributions. Identify which clade
pairs show the largest convergence differential between high-vol and low-vol
bins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from thesis._lib import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ecospace",
        default="thesis/convergence/output/ecospace_genus_mapping.csv",
    )
    ap.add_argument(
        "--pbdb",
        default="data/processed/merged_occurrences.parquet",
    )
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_clade_decomposition")
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()

    eco = pd.read_csv(args.ecospace)
    eco = eco.dropna(subset=["genus", "role_id"]).copy()
    eco = eco[eco["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()

    try:
        occ = pd.read_parquet(
            args.pbdb,
            columns=["source_db", "mid_ma", "genus", "class"],
        )
        occ = occ[occ["source_db"] == "PBDB"].copy()
    except FileNotFoundError:
        print(f"Occurrence file not found: {args.pbdb}")
        return

    occ["genus"] = occ["genus"].astype(str).str.strip()
    occ = occ.dropna(subset=["genus", "mid_ma", "class"]).copy()
    occ["time_bin"] = (pd.to_numeric(occ["mid_ma"], errors="coerce") / 10.0).round() * 10.0

    genus_role = dict(zip(eco["genus"], eco["role_id"]))
    occ["role_id"] = occ["genus"].map(genus_role)
    occ = occ.dropna(subset=["role_id"]).copy()
    occ["clade"] = occ["class"].astype(str).str.strip()

    # For each bin, compute what fraction of genera in each role come from each clade.
    # Then aggregate by clade pair to see cross-clade functional overlap.

    # Genus-level: unique (time_bin, genus, role, clade).
    genus_tb = occ.drop_duplicates(subset=["time_bin", "genus"])[["time_bin", "genus", "role_id", "clade"]].copy()

    vol_map = dict(zip(bins["time_bin"], bins["delta_from_prev_T_field_meanabs"]))
    valid_bins = set(bins["time_bin"])
    genus_tb = genus_tb[genus_tb["time_bin"].isin(valid_bins)].copy()
    genus_tb["vol"] = genus_tb["time_bin"].map(vol_map)

    vol_median = float(genus_tb["vol"].median())
    genus_tb["vol_group"] = np.where(genus_tb["vol"] >= vol_median, "high", "low")

    # For each vol group, compute role distribution by clade.
    clade_role = genus_tb.groupby(["vol_group", "clade", "role_id"])["genus"].nunique().reset_index(name="n_genera")
    clade_totals = clade_role.groupby(["vol_group", "clade"])["n_genera"].sum().rename("clade_total")
    clade_role = clade_role.merge(clade_totals, on=["vol_group", "clade"])
    clade_role["frac"] = clade_role["n_genera"] / clade_role["clade_total"]

    # Top clades by total genera.
    top_clades = genus_tb.groupby("clade")["genus"].nunique().nlargest(15).index.tolist()
    clade_role_top = clade_role[clade_role["clade"].isin(top_clades)].copy()

    # For each clade pair, compute JS-style overlap of role distributions.
    from scipy.spatial.distance import jensenshannon

    all_roles = sorted(clade_role_top["role_id"].unique())
    role_idx = {r: i for i, r in enumerate(all_roles)}

    def _role_vec(df_sub: pd.DataFrame) -> np.ndarray:
        vec = np.zeros(len(all_roles))
        for _, row in df_sub.iterrows():
            vec[role_idx[row["role_id"]]] = row["frac"]
        return vec

    results_pairs = []
    for vol_g in ["high", "low"]:
        sub = clade_role_top[clade_role_top["vol_group"] == vol_g]
        vecs = {}
        for clade in top_clades:
            cs = sub[sub["clade"] == clade]
            if len(cs) > 0:
                vecs[clade] = _role_vec(cs)

        for i, c1 in enumerate(top_clades):
            for c2 in top_clades[i + 1:]:
                if c1 in vecs and c2 in vecs:
                    v1, v2 = vecs[c1], vecs[c2]
                    if v1.sum() > 0 and v2.sum() > 0:
                        sim = 1.0 - float(jensenshannon(v1, v2))
                        results_pairs.append({
                            "vol_group": vol_g,
                            "clade_a": c1,
                            "clade_b": c2,
                            "js_similarity": sim,
                        })

    pairs_df = pd.DataFrame(results_pairs)
    if pairs_df.empty:
        print("No clade pairs to compare.")
        return

    pivot = pairs_df.pivot_table(
        index=["clade_a", "clade_b"], columns="vol_group", values="js_similarity"
    ).dropna()
    pivot["diff_high_minus_low"] = pivot["high"] - pivot["low"]
    pivot = pivot.sort_values("diff_high_minus_low", ascending=False)

    results = {
        "n_clades": len(top_clades),
        "n_pairs": len(pivot),
        "vol_median": vol_median,
        "top_converging_pairs": [
            {"clade_a": idx[0], "clade_b": idx[1], "diff": float(row["diff_high_minus_low"]),
             "high_sim": float(row["high"]), "low_sim": float(row["low"])}
            for idx, row in pivot.head(10).iterrows()
        ],
        "top_diverging_pairs": [
            {"clade_a": idx[0], "clade_b": idx[1], "diff": float(row["diff_high_minus_low"]),
             "high_sim": float(row["high"]), "low_sim": float(row["low"])}
            for idx, row in pivot.tail(10).iterrows()
        ],
    }

    pivot.to_csv(out_dir / "clade_pair_decomposition.csv")
    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Figure.
    top_n = min(20, len(pivot))
    top = pivot.head(top_n // 2)
    bottom = pivot.tail(top_n // 2)
    combined = pd.concat([top, bottom])

    fig, ax = plt.subplots(figsize=(8, max(5, len(combined) * 0.35)))
    labels = [f"{a} × {b}" for a, b in combined.index]
    colors = ["#d62728" if d > 0 else "#2ca02c" for d in combined["diff_high_minus_low"]]
    ax.barh(range(len(combined)), combined["diff_high_minus_low"].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δ JS similarity (high-vol − low-vol)")
    ax.set_title("Clade-pair functional convergence decomposition")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "clade_decomposition.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Clade decomposition",
        "",
        f"- {results['n_clades']} clades, {results['n_pairs']} clade pairs",
        f"- Volatility median split at {results['vol_median']:.4f}",
        "",
        "## Clade pairs that CONVERGE more under high volatility",
    ]
    for p in results["top_converging_pairs"]:
        lines.append(f"- {p['clade_a']} × {p['clade_b']}: Δ = {p['diff']:+.4f} (high={p['high_sim']:.3f}, low={p['low_sim']:.3f})")
    lines.extend(["", "## Clade pairs that DIVERGE more under high volatility"])
    for p in results["top_diverging_pairs"]:
        lines.append(f"- {p['clade_a']} × {p['clade_b']}: Δ = {p['diff']:+.4f} (high={p['high_sim']:.3f}, low={p['low_sim']:.3f})")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
