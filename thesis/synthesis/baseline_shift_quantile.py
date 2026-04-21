"""Non-parametric evidence for the baseline-shift mechanism.

Two tests:
(a) Cumulative OLS of func_sim_js on vol_z restricted to pairs at or below the
    10th/25th/50th/75th taxsim quantile thresholds. If the vol_z coefficient is
    positive at the lowest quantile thresholds (where pairs share almost no
    genera), volatility lifts functional similarity at the floor — consistent
    with the baseline-shift interpretation.
(b) Stratified comparison of mean func_sim for pairs with taxsim < 0.1 across
    volatility tertiles, with bin-cluster bootstrap CIs.

If both agree that volatility lifts functional similarity at the LOW-taxsim end,
the baseline-shift claim is supported without a linearity assumption.
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
from thesis._lib import ensure_dir, z_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairwise",
        default="thesis/convergence/output/pairwise_sample.csv",
    )
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_baseline_shift_quantile")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    pw = pd.read_csv(args.pairwise)
    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["delta_from_prev_T_field_meanabs"]).copy()
    vol_map = dict(zip(bins["time_bin"], z_score(bins["delta_from_prev_T_field_meanabs"].to_numpy())))

    pw["vol_z"] = pw["time_bin"].map(vol_map)
    pw = pw.dropna(subset=["vol_z", "functional_similarity_js", "taxonomic_similarity"]).copy()
    pw["taxsim"] = pw["taxonomic_similarity"].astype(float)

    results: dict[str, object] = {"n_pairs": len(pw)}

    # -------------------------------------------------------
    # (a) Cumulative OLS at taxsim quantile thresholds
    # -------------------------------------------------------
    taxsim_quantiles = [0.10, 0.25, 0.50, 0.75]
    qr_results = {}

    for q in taxsim_quantiles:
        thresh = float(np.quantile(pw["taxsim"], q))
        sub = pw[pw["taxsim"] <= thresh + 1e-9].copy()
        if len(sub) < 50:
            qr_results[f"taxsim_le_q{int(q*100)}"] = {"error": f"too few pairs ({len(sub)})"}
            continue

        y = sub["functional_similarity_js"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(sub)), sub["vol_z"].to_numpy(dtype=float)])

        # OLS on the cumulative subset (all pairs at or below this taxsim threshold).
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        qr_results[f"taxsim_le_q{int(q*100)}"] = {
            "taxsim_threshold": thresh,
            "n_pairs": len(sub),
            "vol_z_beta": float(beta[1]),
            "intercept": float(beta[0]),
        }

    results["cumulative_ols"] = qr_results

    # -------------------------------------------------------
    # (b) Stratified comparison: low-taxsim pairs across volatility tertiles
    # -------------------------------------------------------
    low_tax = pw[pw["taxsim"] < 0.10].copy()
    results["low_taxsim_n_pairs"] = len(low_tax)

    if len(low_tax) >= 30:
        vol_z_vals = low_tax["vol_z"].to_numpy(dtype=float)
        vol_tertiles = np.quantile(vol_z_vals, [1 / 3, 2 / 3])
        low_tax["vol_tertile"] = np.where(
            vol_z_vals <= vol_tertiles[0], "low",
            np.where(vol_z_vals <= vol_tertiles[1], "mid", "high")
        )

        tertile_means = {}
        for t in ["low", "mid", "high"]:
            vals = low_tax.loc[low_tax["vol_tertile"] == t, "functional_similarity_js"]
            tertile_means[t] = {"mean": float(vals.mean()), "n": len(vals)}

        # Bin-cluster bootstrap for the high-low difference.
        diff_obs = tertile_means["high"]["mean"] - tertile_means["low"]["mean"]
        rng = np.random.default_rng(args.seed)
        unique_bins = low_tax["time_bin"].unique()
        diffs = []
        for _ in range(args.n_boot):
            boot_bins = rng.choice(unique_bins, size=len(unique_bins), replace=True)
            boot_df = pd.concat([low_tax[low_tax["time_bin"] == b] for b in boot_bins], ignore_index=True)
            if boot_df.empty:
                continue
            h = boot_df.loc[boot_df["vol_tertile"] == "high", "functional_similarity_js"]
            lo = boot_df.loc[boot_df["vol_tertile"] == "low", "functional_similarity_js"]
            if len(h) > 0 and len(lo) > 0:
                diffs.append(float(h.mean() - lo.mean()))
        diffs_arr = np.array(diffs)

        results["stratified_low_taxsim"] = {
            "vol_tertile_thresholds": [float(vol_tertiles[0]), float(vol_tertiles[1])],
            "tertile_means": tertile_means,
            "diff_high_minus_low": diff_obs,
            "boot_ci_025": float(np.percentile(diffs_arr, 2.5)) if len(diffs_arr) > 0 else float("nan"),
            "boot_ci_975": float(np.percentile(diffs_arr, 97.5)) if len(diffs_arr) > 0 else float("nan"),
            "boot_mean": float(np.mean(diffs_arr)) if len(diffs_arr) > 0 else float("nan"),
            "n_boot": len(diffs_arr),
        }

        # Figure.
        fig, ax = plt.subplots(figsize=(6, 4))
        positions = {"low": 0, "mid": 1, "high": 2}
        colors = {"low": "#2ca02c", "mid": "#ff7f0e", "high": "#d62728"}
        for t in ["low", "mid", "high"]:
            vals = low_tax.loc[low_tax["vol_tertile"] == t, "functional_similarity_js"]
            ax.boxplot(
                vals, positions=[positions[t]], widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=colors[t], alpha=0.6),
                medianprops=dict(color="black"),
            )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Low vol", "Mid vol", "High vol"])
        ax.set_ylabel("Functional similarity (JS)")
        ax.set_title(f"Low-taxonomy pairs (taxsim < 0.1, n={len(low_tax)})")
        fig.tight_layout()
        fig.savefig(fig_dir / "low_taxsim_by_volatility.png", dpi=220)
        plt.close(fig)
    else:
        results["stratified_low_taxsim"] = {"error": f"too few low-taxsim pairs ({len(low_tax)})"}

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    lines = [
        "# Baseline-shift mechanism (non-parametric evidence)",
        "",
        "## (a) Cumulative OLS at taxsim quantile thresholds",
    ]
    for k, v in results.get("cumulative_ols", {}).items():
        if "error" in v:
            lines.append(f"- {k}: {v['error']}")
        else:
            lines.append(f"- {k} (threshold={v['taxsim_threshold']:.3f}, n={v['n_pairs']}): vol_z beta = {v['vol_z_beta']:.4f}")

    strat = results.get("stratified_low_taxsim", {})
    lines.extend(["", "## (b) Stratified: low-taxsim pairs across volatility tertiles"])
    if "error" in strat:
        lines.append(f"- {strat['error']}")
    else:
        for t in ["low", "mid", "high"]:
            m = strat["tertile_means"].get(t, {})
            lines.append(f"- {t} volatility: mean func_sim = {m.get('mean', 'nan'):.4f} (n={m.get('n', '?')})")
        lines.append(f"- Diff (high - low): {strat['diff_high_minus_low']:.4f}")
        lines.append(f"- 95% CI (bin-cluster bootstrap): [{strat['boot_ci_025']:.4f}, {strat['boot_ci_975']:.4f}]")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
