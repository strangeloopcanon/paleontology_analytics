"""Characterise PBDB ecospace missingness per time bin.

Produces:
- Coverage rates per bin (fraction of genera with complete ecospace)
- Temporal trend in annotation quality
- Correlation between missingness and the convergence metric
- Supplementary table of per-bin coverage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_name(x: object) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _corr_and_perm(
    x: np.ndarray, y: np.ndarray, *, permutations: int = 10000, seed: int = 42
) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 6:
        return {"corr": float("nan"), "p_perm": float("nan"), "n": n}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    more = sum(
        1 for _ in range(permutations) if abs(float(np.corrcoef(x, rng.permutation(y))[0, 1])) >= abs(obs)
    )
    return {"corr": obs, "p_perm": (more + 1) / (permutations + 1), "n": n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--ecospace-mapping", default="thesis/convergence/output_v3_fullpbdb/ecospace_genus_mapping.csv")
    p.add_argument("--convergence-bins", default="thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv")
    p.add_argument("--out", default="thesis/synthesis/output_ecospace_missingness")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--min-occ-per-genus", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # Load PBDB occurrences.
    occ = pd.read_parquet(
        args.pbdb,
        columns=["source_db", "occurrence_id", "mid_ma", "genus", "phylum", "class"],
    )
    occ = occ[occ["source_db"] == "PBDB"].copy()
    occ["genus"] = occ["genus"].map(_clean_name)
    occ = occ.dropna(subset=["genus", "mid_ma"]).copy()
    occ["time_bin"] = (
        pd.to_numeric(occ["mid_ma"], errors="coerce") / args.time_bin_myr
    ).round() * args.time_bin_myr

    # Load ecospace mapping.
    eco = pd.read_csv(args.ecospace_mapping)
    eco["genus"] = eco["genus"].map(_clean_name)
    eco = eco.dropna(subset=["genus"]).copy()

    # Determine trait completeness per genus.
    eco["has_role"] = eco["role_id"].notna() & (eco["role_id"].astype(str).str.strip() != "")
    eco["has_jev"] = eco["jev"].notna() & (eco["jev"].astype(str).str.strip() != "")
    eco["is_marine"] = eco["jev"].astype(str).str.contains("marine", case=False, na=False)

    genus_traits = eco.set_index("genus")[["has_role", "has_jev", "is_marine"]].to_dict("index")

    # Filter genera by minimum occurrence count.
    genus_counts = occ["genus"].value_counts()
    valid_genera = set(genus_counts[genus_counts >= args.min_occ_per_genus].index)

    # Per-bin coverage.
    rows = []
    for tb, grp in occ.groupby("time_bin"):
        genera_in_bin = set(grp["genus"].unique()) & valid_genera
        n_genera = len(genera_in_bin)
        if n_genera == 0:
            continue

        n_in_ecospace = sum(1 for g in genera_in_bin if g in genus_traits)
        n_has_role = sum(1 for g in genera_in_bin if genus_traits.get(g, {}).get("has_role", False))
        n_marine = sum(1 for g in genera_in_bin if genus_traits.get(g, {}).get("is_marine", False))
        n_marine_with_role = sum(
            1
            for g in genera_in_bin
            if genus_traits.get(g, {}).get("is_marine", False) and genus_traits.get(g, {}).get("has_role", False)
        )

        rows.append(
            {
                "time_bin": float(tb),
                "n_genera_in_bin": n_genera,
                "n_in_ecospace_api": n_in_ecospace,
                "n_has_complete_role": n_has_role,
                "n_marine_annotated": n_marine,
                "n_marine_with_complete_role": n_marine_with_role,
                "frac_in_ecospace": n_in_ecospace / n_genera if n_genera else float("nan"),
                "frac_has_role": n_has_role / n_genera if n_genera else float("nan"),
                "frac_marine": n_marine / n_genera if n_genera else float("nan"),
                "frac_marine_with_role": n_marine_with_role / n_genera if n_genera else float("nan"),
            }
        )

    coverage = pd.DataFrame(rows).sort_values("time_bin", ascending=False).reset_index(drop=True)
    coverage.to_csv(out_dir / "ecospace_coverage_per_bin.csv", index=False)

    # Merge with convergence metrics.
    conv = pd.read_csv(args.convergence_bins)
    merged = conv.merge(coverage, on="time_bin", how="inner")
    merged.to_csv(out_dir / "merged_coverage_convergence.csv", index=False)

    # Correlation tests: does missingness correlate with convergence?
    results: dict[str, object] = {}
    if len(merged) >= 6 and "functional_excess_similarity_js" in merged.columns:
        y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
        for col in ["frac_has_role", "frac_marine_with_role", "frac_in_ecospace"]:
            x = merged[col].to_numpy(dtype=float)
            results[f"corr_{col}_vs_convergence"] = _corr_and_perm(x, y, seed=42)

        # Also test: does coverage correlate with time (secular trend)?
        t = merged["time_bin"].to_numpy(dtype=float)
        for col in ["frac_has_role", "frac_marine_with_role"]:
            x = merged[col].to_numpy(dtype=float)
            results[f"corr_{col}_vs_time"] = _corr_and_perm(x, t, seed=43)

        # CRITICAL: partial correlation of coverage vs convergence, controlling for time.
        # If this is weak, the raw r=0.90 is driven by the shared time trend and is not
        # an independent confound. If strong, coverage may confound the volatility result.
        def _partial_corr(x: np.ndarray, y_: np.ndarray, z: np.ndarray) -> float:
            """Partial correlation of x and y controlling for z (via residualisation)."""
            mask = np.isfinite(x) & np.isfinite(y_) & np.isfinite(z)
            if int(np.sum(mask)) < 6:
                return float("nan")
            A = np.column_stack([np.ones(int(np.sum(mask))), z[mask]])
            bx, *_ = np.linalg.lstsq(A, x[mask], rcond=None)
            by, *_ = np.linalg.lstsq(A, y_[mask], rcond=None)
            rx, ry_ = x[mask] - A.dot(bx), y_[mask] - A.dot(by)
            return float(np.corrcoef(rx, ry_)[0, 1])

        for col in ["frac_has_role", "frac_marine_with_role", "frac_in_ecospace"]:
            x = merged[col].to_numpy(dtype=float)
            r_partial = _partial_corr(x, y, t)
            results[f"partial_corr_{col}_vs_convergence_controlling_time"] = {
                "partial_r": r_partial,
                "n": int(np.sum(np.isfinite(x) & np.isfinite(y) & np.isfinite(t))),
            }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Figures.
    if len(coverage) > 5:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        tb = coverage["time_bin"]

        axes[0].plot(tb, coverage["frac_has_role"], "o-", label="Has complete role (all genera)", color="#1f77b4")
        axes[0].plot(
            tb, coverage["frac_marine_with_role"], "s-", label="Marine + complete role", color="#d62728"
        )
        axes[0].set_ylabel("Fraction of genera")
        axes[0].legend(loc="lower left")
        axes[0].set_title("PBDB ecospace annotation completeness per 10 Myr bin")
        axes[0].invert_xaxis()

        axes[1].bar(tb, coverage["n_genera_in_bin"], width=7, alpha=0.5, color="#999999", label="Total genera")
        axes[1].bar(
            tb,
            coverage["n_marine_with_complete_role"],
            width=7,
            alpha=0.8,
            color="#d62728",
            label="Marine + complete role",
        )
        axes[1].set_xlabel("Time bin (Ma)")
        axes[1].set_ylabel("Number of genera")
        axes[1].legend()
        axes[1].invert_xaxis()

        fig.tight_layout()
        fig.savefig(fig_dir / "ecospace_coverage_timeseries.png", dpi=220)
        plt.close(fig)

    # If convergence data available, scatter coverage vs convergence.
    if len(merged) > 5 and "functional_excess_similarity_js" in merged.columns:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(
            merged["frac_marine_with_role"],
            merged["functional_excess_similarity_js"],
            s=50,
            alpha=0.8,
            color="#1f77b4",
        )
        ax.set_xlabel("Fraction of genera: marine + complete ecospace role")
        ax.set_ylabel("Functional excess similarity (JS residual)")
        ax.set_title("Does ecospace annotation quality confound convergence?")
        fig.tight_layout()
        fig.savefig(fig_dir / "coverage_vs_convergence.png", dpi=220)
        plt.close(fig)

    # Summary.
    lines = [
        "# Ecospace missingness diagnostic",
        "",
        f"- Bins analysed: {len(coverage)}",
        f"- Mean fraction with complete role (all genera): {coverage['frac_has_role'].mean():.3f}",
        f"- Mean fraction marine + complete role: {coverage['frac_marine_with_role'].mean():.3f}",
        "",
        "## Temporal trend in annotation quality",
        "",
    ]
    for key, val in results.items():
        if key.startswith("partial_corr_"):
            continue  # reported in its own section below
        if isinstance(val, dict) and "corr" in val:
            lines.append(f"- {key}: corr={val['corr']:.3f}, p={val.get('p_perm', 'nan'):.3g}, n={val.get('n', 0)}")
        elif isinstance(val, dict):
            lines.append(f"- {key}: {val}")
        else:
            lines.append(f"- {key}: {val}")

    lines.append("")
    lines.append("## Partial correlations (controlling for time)")
    lines.append("")
    for key, val in results.items():
        if key.startswith("partial_corr_") and isinstance(val, dict):
            lines.append(f"- {key}: partial_r={val.get('partial_r', 'nan'):.3f}, n={val.get('n', 0)}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the **partial** correlation of coverage vs convergence (controlling for time) is weak,",
            "then the raw r=0.90 is driven by the shared secular trend and coverage is not an independent",
            "confound. If the partial correlation is still strong, coverage may confound the volatility result.",
            "",
            "## Files",
            "",
            f"- Coverage table: `{out_dir / 'ecospace_coverage_per_bin.csv'}`",
            f"- Merged table: `{out_dir / 'merged_coverage_convergence.csv'}`",
            f"- Stats: `{out_dir / 'analysis_results.json'}`",
            f"- Figures: `{fig_dir}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
