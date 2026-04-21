"""Spatial null test for the volatility-convergence association.

Within each time bin, permutes the functional_similarity_js values across
locality pairs while keeping taxonomic_similarity fixed. This breaks the
geographic coupling between functional and taxonomic similarity — i.e., the
spatial structure of ecological roles — while preserving within-bin marginal
distributions.

Uses the OBSERVED global OLS slope (func ~ tax) to compute residuals under
each permutation, then averages per bin to produce a null functional excess
time series. Tests whether the partial correlation of excess vs volatility
exceeds the null distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis._lib import build_controls, ensure_dir, residualize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument(
        "--pairwise",
        default="thesis/convergence/output/pairwise_sample.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_spatial_null")
    ap.add_argument("--n-perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    pw = pd.read_csv(args.pairwise)

    y_obs = bins["functional_excess_similarity_js"].to_numpy(dtype=float)
    v = bins["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    controls = build_controls(bins)

    rv = residualize(v, controls)
    ry_obs = residualize(y_obs, controls)
    mask = np.isfinite(rv) & np.isfinite(ry_obs)
    if mask.sum() < 6:
        print("Too few valid bins; aborting.")
        return
    obs_corr = float(np.corrcoef(rv[mask], ry_obs[mask])[0, 1])

    # Global OLS slope from observed data (fixed across permutations).
    x_tax = pw["taxonomic_similarity"].to_numpy(dtype=float)
    y_func = pw["functional_similarity_js"].to_numpy(dtype=float)
    ok = np.isfinite(x_tax) & np.isfinite(y_func)
    A = np.column_stack([np.ones(ok.sum()), x_tax[ok]])
    beta, *_ = np.linalg.lstsq(A, y_func[ok], rcond=None)
    global_intercept, global_slope = float(beta[0]), float(beta[1])

    bin_order = bins["time_bin"].to_numpy(dtype=float)

    rng = np.random.default_rng(args.seed)
    null_corrs = np.full(args.n_perms, np.nan)

    for rep in range(args.n_perms):
        excess_null = np.full(len(bin_order), np.nan)
        for i, tb in enumerate(bin_order):
            sub = pw[pw["time_bin"] == tb]
            if len(sub) < 10:
                continue
            func_vals = sub["functional_similarity_js"].to_numpy(dtype=float)
            tax_vals = sub["taxonomic_similarity"].to_numpy(dtype=float)
            # Permute functional similarities within this bin.
            func_perm = rng.permutation(func_vals)
            pred = global_intercept + global_slope * tax_vals
            residuals = func_perm - pred
            excess_null[i] = float(np.nanmean(residuals))

        ry_null = residualize(excess_null, controls)
        m = np.isfinite(rv) & np.isfinite(ry_null)
        if m.sum() < 6:
            continue
        null_corrs[rep] = float(np.corrcoef(rv[m], ry_null[m])[0, 1])

    valid = np.isfinite(null_corrs)
    n_valid = int(valid.sum())
    more = int(np.sum(np.abs(null_corrs[valid]) >= abs(obs_corr)))
    p_spatial = (more + 1) / (n_valid + 1)

    results = {
        "observed_partial_r": obs_corr,
        "n_perms": n_valid,
        "p_spatial_null": p_spatial,
        "null_mean": float(np.nanmean(null_corrs)),
        "null_sd": float(np.nanstd(null_corrs)),
        "null_95th": float(np.nanpercentile(null_corrs[valid], 95)) if n_valid > 0 else float("nan"),
        "null_99th": float(np.nanpercentile(null_corrs[valid], 99)) if n_valid > 0 else float("nan"),
        "controls": "time + sampling_PCA_PC12 + provinciality",
        "global_ols_slope": global_slope,
        "global_ols_intercept": global_intercept,
    }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# Spatial null test",
        "",
        "Within-bin permutation of functional similarity values across locality pairs,",
        "keeping taxonomic similarity fixed. Tests whether the geographic coupling of",
        "roles to taxonomy drives the volatility-convergence association.",
        "",
        f"- Observed partial r: {obs_corr:.4f}",
        f"- Spatial null p: {p_spatial:.4f} ({n_valid} valid permutations)",
        f"- Null distribution: mean={results['null_mean']:.4f}, sd={results['null_sd']:.4f}",
        f"- Null 95th percentile: {results['null_95th']:.4f}",
        f"- Null 99th percentile: {results['null_99th']:.4f}",
        "",
        f"Interpretation: {'observed r exceeds spatial null' if p_spatial < 0.05 else 'observed r does NOT exceed spatial null'}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
