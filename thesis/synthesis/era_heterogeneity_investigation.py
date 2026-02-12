"""Investigate why the volatility-convergence signal concentrates in the Mesozoic.

Tests competing explanations:
1. Volatility amplitude: do Mesozoic bins simply have larger T swings?
2. Ecospace saturation: is Paleozoic ecospace coverage/diversity too low?
3. Paleogeographic connectivity: does land fragmentation differ?
4. Sampling structure: does Paleozoic trait annotation coverage differ?

Produces era-split results and a decomposition of the era effect.
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


def _z(x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if int(np.sum(mask)) < 3:
        return out
    mu, sd = float(np.mean(x[mask])), float(np.std(x[mask], ddof=1))
    sd = sd if sd > 0 else 1.0
    out[mask] = (x[mask] - mu) / sd
    return out


def _partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if int(np.sum(mask)) < 8:
        return float("nan")
    xm, ym, cm = x[mask], y[mask], controls[mask]
    A = np.column_stack([np.ones(len(cm)), cm])
    bx, *_ = np.linalg.lstsq(A, xm, rcond=None)
    by, *_ = np.linalg.lstsq(A, ym, rcond=None)
    rx, ry = xm - A.dot(bx), ym - A.dot(by)
    return float(np.corrcoef(rx, ry)[0, 1])


def _perm_p(x: np.ndarray, y: np.ndarray, *, perm: int = 10000, seed: int = 42) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 6:
        return {"corr": float("nan"), "p": float("nan"), "n": n}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    more = sum(1 for _ in range(perm) if abs(float(np.corrcoef(x, rng.permutation(y))[0, 1])) >= abs(obs))
    return {"corr": obs, "p": (more + 1) / (perm + 1), "n": n}


ERA_BOUNDARIES = {"Paleozoic": (540, 252), "Mesozoic": (252, 66), "Cenozoic": (66, 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv",
    )
    ap.add_argument(
        "--coverage",
        default="thesis/synthesis/output_ecospace_missingness/ecospace_coverage_per_bin.csv",
        help="Ecospace coverage table (from missingness diagnostic).",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_era_heterogeneity")
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    # Assign eras.
    def _era(tb: float) -> str:
        for name, (lo, hi) in ERA_BOUNDARIES.items():
            if hi <= tb <= lo:
                return name
        return "unknown"

    bins["era"] = bins["time_bin"].map(_era)

    # Merge coverage if available.
    cov_path = Path(args.coverage)
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        bins = bins.merge(cov[["time_bin", "frac_marine_with_role", "n_marine_with_complete_role"]], on="time_bin", how="left")

    y = bins["functional_excess_similarity_js"].to_numpy(dtype=float)
    v = bins["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    t = bins["time_bin"].to_numpy(dtype=float)

    results: dict[str, object] = {"n_bins": len(bins)}

    # -------------------------------------------------------
    # 1) Full-sample and per-era correlations (raw and partial on time)
    # -------------------------------------------------------
    results["full_sample_raw"] = _perm_p(v, y, seed=args.seed)
    results["full_sample_partial_time"] = {
        "corr": _partial_corr(v, y, t.reshape(-1, 1)),
    }

    era_results = {}
    for era_name in ["Paleozoic", "Mesozoic", "Cenozoic"]:
        mask = bins["era"] == era_name
        n_era = int(mask.sum())
        if n_era < 6:
            era_results[era_name] = {"n": n_era, "note": "too few bins"}
            continue
        ve, ye, te = v[mask], y[mask], t[mask]
        era_results[era_name] = {
            "n": n_era,
            "raw_corr": _perm_p(ve, ye, seed=args.seed),
            "partial_corr_time": _partial_corr(ve, ye, te.reshape(-1, 1)),
            "mean_volatility": float(np.nanmean(ve)),
            "mean_convergence": float(np.nanmean(ye)),
        }
        # Add coverage stats if available.
        if "frac_marine_with_role" in bins.columns:
            era_results[era_name]["mean_ecospace_coverage"] = float(
                bins.loc[mask, "frac_marine_with_role"].mean()
            )
    results["per_era"] = era_results

    # -------------------------------------------------------
    # 2) Explanation 1: volatility amplitude
    # -------------------------------------------------------
    vol_by_era = bins.groupby("era")["delta_from_prev_T_field_meanabs"].agg(["mean", "std", "count"]).to_dict("index")
    results["volatility_amplitude_by_era"] = vol_by_era

    # -------------------------------------------------------
    # 3) Explanation 2: ecospace saturation (n unique roles per bin)
    # -------------------------------------------------------
    if "n_localities" in bins.columns:
        loc_by_era = bins.groupby("era")["n_localities"].agg(["mean", "std"]).to_dict("index")
        results["locality_richness_by_era"] = loc_by_era

    # -------------------------------------------------------
    # 4) Explanation 3: paleogeographic connectivity
    # -------------------------------------------------------
    for col in ["land_area_fraction", "land_components", "coastline_index"]:
        if col in bins.columns:
            geo_by_era = bins.groupby("era")[col].agg(["mean", "std"]).to_dict("index")
            results[f"{col}_by_era"] = geo_by_era

    # -------------------------------------------------------
    # 5) Explanation 4: sampling / annotation coverage
    # -------------------------------------------------------
    if "frac_marine_with_role" in bins.columns:
        cov_by_era = bins.groupby("era")["frac_marine_with_role"].agg(["mean", "std"]).to_dict("index")
        results["ecospace_coverage_by_era"] = cov_by_era

    # -------------------------------------------------------
    # Figure: era comparison (3-panel)
    # -------------------------------------------------------
    era_colors = {"Paleozoic": "#2ca02c", "Mesozoic": "#d62728", "Cenozoic": "#1f77b4"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: volatility vs convergence, colored by era.
    ax = axes[0]
    for era_name, color in era_colors.items():
        mask = bins["era"] == era_name
        if mask.sum() < 2:
            continue
        ax.scatter(v[mask], y[mask], c=color, label=era_name, s=50, alpha=0.8, edgecolors="none")
        # Fit line per era.
        vm, ym = v[mask], y[mask]
        valid = np.isfinite(vm) & np.isfinite(ym)
        if valid.sum() >= 4:
            A = np.vstack([vm[valid], np.ones(valid.sum())]).T
            coef, *_ = np.linalg.lstsq(A, ym[valid], rcond=None)
            xx = np.linspace(float(vm[valid].min()), float(vm[valid].max()), 30)
            ax.plot(xx, coef[0] * xx + coef[1], color=color, linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Climate volatility (|dT| field mean)")
    ax.set_ylabel("Functional excess similarity")
    ax.set_title("Volatility vs convergence by era")
    ax.legend()

    # Panel 2: volatility distribution by era.
    ax = axes[1]
    era_vols = {e: v[bins["era"] == e] for e in era_colors}
    positions = list(range(len(era_colors)))
    for i, (era_name, color) in enumerate(era_colors.items()):
        vals = era_vols[era_name]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                       boxprops={"facecolor": color, "alpha": 0.6},
                       medianprops={"color": "black"})
    ax.set_xticks(positions)
    ax.set_xticklabels(list(era_colors.keys()))
    ax.set_ylabel("Climate volatility")
    ax.set_title("Volatility amplitude by era")

    # Panel 3: ecospace coverage by era (if available).
    ax = axes[2]
    if "frac_marine_with_role" in bins.columns:
        for i, (era_name, color) in enumerate(era_colors.items()):
            mask = bins["era"] == era_name
            vals = bins.loc[mask, "frac_marine_with_role"].dropna().to_numpy()
            if len(vals) > 0:
                ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                           boxprops={"facecolor": color, "alpha": 0.6},
                           medianprops={"color": "black"})
        ax.set_xticks(positions)
        ax.set_xticklabels(list(era_colors.keys()))
        ax.set_ylabel("Ecospace coverage (marine + complete role)")
        ax.set_title("Annotation quality by era")
    else:
        ax.text(0.5, 0.5, "Coverage data not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Ecospace coverage by era")

    fig.tight_layout()
    fig.savefig(fig_dir / "era_comparison.png", dpi=220)
    plt.close(fig)

    # -------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------
    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Summary.
    lines = [
        "# Era heterogeneity investigation",
        "",
        "## Per-era volatility-convergence correlations",
        "",
    ]
    for era_name in ["Paleozoic", "Mesozoic", "Cenozoic"]:
        er = era_results.get(era_name, {})
        if "note" in er:
            lines.append(f"- {era_name}: {er['note']} (n={er.get('n', 0)})")
        else:
            raw = er.get("raw_corr", {})
            lines.append(
                f"- {era_name} (n={er.get('n', 0)}): raw corr={raw.get('corr', 'nan'):.3f}, "
                f"perm-p={raw.get('p', 'nan'):.3g}; "
                f"partial|time={er.get('partial_corr_time', 'nan'):.3f}; "
                f"mean vol={er.get('mean_volatility', 'nan'):.3f}"
            )

    lines.extend([
        "",
        "## Competing explanations",
        "",
        "### Volatility amplitude",
    ])
    for era_name, vals in results.get("volatility_amplitude_by_era", {}).items():
        lines.append(f"- {era_name}: mean={vals.get('mean', 'nan'):.3f}, sd={vals.get('std', 'nan'):.3f}")

    if "ecospace_coverage_by_era" in results:
        lines.extend(["", "### Ecospace annotation coverage"])
        for era_name, vals in results["ecospace_coverage_by_era"].items():
            lines.append(f"- {era_name}: mean={vals.get('mean', 'nan'):.3f}, sd={vals.get('std', 'nan'):.3f}")

    for geo_col in ["land_area_fraction", "land_components", "coastline_index"]:
        key = f"{geo_col}_by_era"
        if key in results:
            lines.extend(["", f"### {geo_col}"])
            for era_name, vals in results[key].items():
                lines.append(f"- {era_name}: mean={vals.get('mean', 'nan'):.3f}")

    lines.extend(["", "## Files", "",
                   f"- Stats: `{out_dir / 'analysis_results.json'}`",
                   f"- Figures: `{fig_dir}`"])

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
