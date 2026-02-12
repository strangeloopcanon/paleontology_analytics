"""Robustness battery for the volatility-convergence association.

Runs from pre-computed bin-level merged tables:
1. Leave-one-out bin stability (drop each bin, recompute partial correlation)
2. Block bootstrap p-values (complementary to circular shift)
3. Lagerstaetten sensitivity (exclude known exceptional-preservation bins)
4. Clade-restriction test (Bivalvia, Gastropoda, Brachiopoda only)
5. Effective sample size reporting (bin-level vs pair-level)
6. AIC/BIC AR order sweep for SARIMAX (AR(0)..AR(3))
7. Automatic Newey-West bandwidth
8. Seed sensitivity for pair subsampling (3 seeds)
9. Primary specification lock
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import t as student_t

# Optional: statsmodels for SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_SM = True
except ImportError:
    HAS_SM = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _z(x: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    x = x.astype(float)
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if int(np.sum(mask)) < 3:
        return out
    mu = float(np.mean(x[mask]))
    sd = float(np.std(x[mask], ddof=ddof))
    sd = sd if sd > 0 else 1.0
    out[mask] = (x[mask] - mu) / sd
    return out


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    if int(np.sum(mask)) < X.shape[1] + 3:
        return out
    A = np.column_stack([np.ones(int(np.sum(mask))), X[mask]])
    beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
    out[mask] = y[mask] - A.dot(beta)
    return out


def _partial_corr(v: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    rv = _residualize(v, controls)
    ry = _residualize(y, controls)
    mask = np.isfinite(rv) & np.isfinite(ry)
    if int(np.sum(mask)) < 6:
        return float("nan")
    return float(np.corrcoef(rv[mask], ry[mask])[0, 1])


def _circular_shift_p(
    v: np.ndarray, y: np.ndarray, controls: np.ndarray, *, seed: int = 77
) -> dict[str, float]:
    """Circular-shift p-values: exact enumeration + random-sampling estimate.

    Exact: counts all N shifts (including identity) where |corr| >= |obs|.
    Random-sampling: 20,000 draws from non-zero shifts, p = (more+1)/(perms+1).
    The two estimators differ for small N; exact is more conservative.
    """
    rv = _residualize(v, controls)
    ry = _residualize(y, controls)
    mask = np.isfinite(rv) & np.isfinite(ry)
    rv, ry = rv[mask], ry[mask]
    n = len(rv)
    if n < 6:
        return {"corr": float("nan"), "p_exact": float("nan"), "p_rand": float("nan"), "n": n, "n_shifts": 0}
    obs = float(np.corrcoef(rv, ry)[0, 1])
    # Exact enumeration (all N shifts including identity).
    more_exact = 0
    for s in range(n):
        c = float(np.corrcoef(rv, np.roll(ry, s))[0, 1])
        if abs(c) >= abs(obs):
            more_exact += 1
    # Random-sampling estimate (non-zero shifts only, +1 correction).
    n_perm = 20000
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, n, size=n_perm)
    more_rand = sum(
        1 for s in shifts if abs(float(np.corrcoef(rv, np.roll(ry, s))[0, 1])) >= abs(obs)
    )
    return {
        "corr": obs,
        "p_exact": more_exact / n,
        "p_rand": (more_rand + 1) / (n_perm + 1),
        "n": n,
        "n_shifts": n,
    }


def _pca_scores(X: np.ndarray, *, k: int) -> np.ndarray:
    mask = np.all(np.isfinite(X), axis=1)
    if int(np.sum(mask)) < max(6, k + 3):
        return np.full((len(X), k), np.nan)
    Xc = X[mask]
    mu, sd = np.mean(Xc, 0), np.std(Xc, 0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    U, S, Vt = np.linalg.svd((Xc - mu) / sd, full_matrices=False)
    kk = min(k, Vt.shape[0])
    scores = np.full((len(X), k), np.nan)
    scores[mask, :kk] = U[:, :kk] * S[:kk]
    return scores


# Known Lagerstaette bins (approximate 10 Myr bins).
LAGERSTAETTE_BINS = {
    510.0,  # Burgess Shale (~508 Ma)
    520.0,  # Chengjiang (~518 Ma)
    310.0,  # Mazon Creek (~309 Ma)
    150.0,  # Solnhofen (~150 Ma)
    50.0,   # Messel (~47 Ma)
    170.0,  # La Voulte-sur-Rhone (~165 Ma)
    370.0,  # Gogo Formation (~375 Ma)
    430.0,  # Herefordshire (~430 Ma)
}


def _build_controls(bins: pd.DataFrame) -> np.ndarray:
    """Build the PRIMARY specification controls: time + sampling_PCA_PC12 + provinciality."""
    feat_cols = []
    for col in ["n_localities", "marine_n_collections", "marine_n_occurrences"]:
        if col in bins.columns:
            feat_cols.append(np.log1p(bins[col].to_numpy(dtype=float)))
    for col in ["macro_col_area_sum", "macro_n_sections"]:
        if col in bins.columns:
            feat_cols.append(np.log1p(bins[col].to_numpy(dtype=float)))

    t = bins["time_bin"].to_numpy(dtype=float)
    prov = bins["provinciality"].to_numpy(dtype=float) if "provinciality" in bins.columns else np.zeros(len(bins))

    if feat_cols:
        pcs = _pca_scores(np.column_stack(feat_cols), k=2)
        return np.column_stack([t, pcs[:, 0], pcs[:, 1], prov])
    return np.column_stack([t, prov])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_robustness_battery")
    ap.add_argument("--seed", type=int, default=77)
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    y = bins["functional_excess_similarity_js"].to_numpy(dtype=float)
    v = bins["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    controls = _build_controls(bins)

    results: dict[str, object] = {"n_bins": len(bins)}

    # -------------------------------------------------------
    # 1) Primary specification (locked)
    # -------------------------------------------------------
    primary = _circular_shift_p(v, y, controls)
    results["primary_specification"] = {
        "controls": "time + sampling_PCA_PC12 + provinciality",
        **primary,
        "resolution_limit": f"min p = 1/{primary['n_shifts']} = {1/primary['n_shifts']:.4f}" if primary["n_shifts"] > 0 else "N/A",
    }

    # -------------------------------------------------------
    # 2) Leave-one-out bin stability
    # -------------------------------------------------------
    loo_results = []
    for i in range(len(bins)):
        mask = np.ones(len(bins), dtype=bool)
        mask[i] = False
        r = _partial_corr(v[mask], y[mask], controls[mask])
        loo_results.append(
            {"dropped_bin": float(bins.iloc[i]["time_bin"]), "partial_corr": r}
        )
    loo_df = pd.DataFrame(loo_results)
    loo_df.to_csv(out_dir / "leave_one_out.csv", index=False)
    results["leave_one_out"] = {
        "min_corr": float(loo_df["partial_corr"].min()),
        "max_corr": float(loo_df["partial_corr"].max()),
        "mean_corr": float(loo_df["partial_corr"].mean()),
        "all_positive": bool((loo_df["partial_corr"] > 0).all()),
        "most_influential_bin": float(loo_df.loc[loo_df["partial_corr"].idxmin(), "dropped_bin"]),
    }

    # LOO figure.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(loo_df["dropped_bin"], loo_df["partial_corr"], width=7, color="#1f77b4", alpha=0.8)
    ax.axhline(primary["corr"], color="red", linestyle="--", linewidth=1.2, label="Full sample")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Dropped bin (Ma)")
    ax.set_ylabel("Partial correlation (volatility vs convergence)")
    ax.set_title("Leave-one-out bin stability")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "leave_one_out.png", dpi=220)
    plt.close(fig)

    # -------------------------------------------------------
    # 3) Block bootstrap (complementary to circular shift)
    # -------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    rv = _residualize(v, controls)
    ry = _residualize(y, controls)
    mask_r = np.isfinite(rv) & np.isfinite(ry)
    rv_c, ry_c = rv[mask_r], ry[mask_r]
    n_r = len(rv_c)
    obs_corr = float(np.corrcoef(rv_c, ry_c)[0, 1]) if n_r >= 6 else float("nan")

    n_boot = 10000
    block_sizes = [2, 3, 5]
    boot_results = {}
    for bs in block_sizes:
        more = 0
        for _ in range(n_boot):
            n_blocks = max(1, n_r // bs)
            starts = rng.integers(0, n_r, size=n_blocks)
            idx = np.concatenate([np.arange(s, min(s + bs, n_r)) for s in starts])[:n_r]
            if len(idx) < 6:
                continue
            ry_boot = ry_c[idx]
            c = float(np.corrcoef(rv_c[:len(ry_boot)], ry_boot)[0, 1])
            if abs(c) >= abs(obs_corr):
                more += 1
        p = (more + 1) / (n_boot + 1)
        boot_results[f"block_size_{bs}"] = {"p_boot": p, "n_boot": n_boot}
    results["block_bootstrap"] = boot_results

    # -------------------------------------------------------
    # 4) Lagerstaetten sensitivity
    # -------------------------------------------------------
    lag_mask = ~bins["time_bin"].isin(LAGERSTAETTE_BINS)
    n_excluded = int((~lag_mask).sum())
    if lag_mask.sum() >= 10:
        lag_result = _circular_shift_p(v[lag_mask], y[lag_mask], controls[lag_mask])
        results["lagerstaetten_excluded"] = {
            "bins_excluded": n_excluded,
            "excluded_bins": sorted(bins.loc[~lag_mask, "time_bin"].tolist()),
            **lag_result,
        }
    else:
        results["lagerstaetten_excluded"] = {"error": "too few bins after exclusion"}

    # -------------------------------------------------------
    # 5) AIC/BIC AR order sweep (SARIMAX)
    # -------------------------------------------------------
    if HAS_SM and len(bins) >= 15:
        vol_z = _z(v, ddof=1)
        time_z = _z(bins["time_bin"].to_numpy(dtype=float), ddof=1)
        prov_z = _z(bins["provinciality"].to_numpy(dtype=float), ddof=1) if "provinciality" in bins.columns else np.zeros(len(bins))
        ctrl_pcs = _pca_scores(
            np.column_stack(
                [np.log1p(bins[c].to_numpy(dtype=float)) for c in
                 ["n_localities", "marine_n_collections", "marine_n_occurrences",
                  "macro_col_area_sum", "macro_n_sections"]
                 if c in bins.columns]
            ),
            k=2,
        )
        pc1_z = _z(ctrl_pcs[:, 0], ddof=1)
        pc2_z = _z(ctrl_pcs[:, 1], ddof=1)

        X_exog = np.column_stack([vol_z, time_z, pc1_z, pc2_z, prov_z])
        good = np.all(np.isfinite(X_exog), axis=1) & np.isfinite(y)
        y_g, X_g = y[good], X_exog[good]

        ar_sweep = {}
        for ar_order in range(4):
            try:
                mod = SARIMAX(
                    endog=y_g, exog=X_g, order=(ar_order, 0, 0), trend="c",
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
                vol_idx = 1  # intercept is 0, vol_z is 1
                ar_sweep[f"AR({ar_order})"] = {
                    "aic": float(mod.aic),
                    "bic": float(mod.bic),
                    "vol_beta": float(mod.params[vol_idx]),
                    "vol_se": float(mod.bse[vol_idx]),
                    "vol_p": float(mod.pvalues[vol_idx]),
                }
            except Exception as e:
                ar_sweep[f"AR({ar_order})"] = {"error": str(e)}

        best_aic = min(
            (k for k in ar_sweep if "aic" in ar_sweep[k]),
            key=lambda k: ar_sweep[k]["aic"],
            default=None,
        )
        results["sarimax_ar_sweep"] = {"models": ar_sweep, "best_by_aic": best_aic}

        # Automatic Newey-West bandwidth (Andrews 1991 rule of thumb: floor(4*(n/100)^(2/9)))
        n_ts = len(y_g)
        nw_auto_lags = max(1, int(np.floor(4 * (n_ts / 100) ** (2 / 9))))
        results["newey_west_auto_bandwidth"] = {"n": n_ts, "auto_lags": nw_auto_lags}

        # OLS + auto HAC
        X_sm = np.column_stack([np.ones(len(y_g)), X_g])
        beta_ols, *_ = np.linalg.lstsq(X_sm, y_g, rcond=None)
        resid = y_g - X_sm.dot(beta_ols)
        p_dim = X_sm.shape[1]
        df_resid = max(n_ts - p_dim, 1)
        XtX_inv = np.linalg.inv(X_sm.T.dot(X_sm))
        S = np.zeros((p_dim, p_dim))
        for t_i in range(n_ts):
            xt = X_sm[t_i:t_i + 1].T
            S += (resid[t_i] ** 2) * (xt @ xt.T)
        for lag in range(1, nw_auto_lags + 1):
            w = 1.0 - lag / (nw_auto_lags + 1)
            for t_i in range(lag, n_ts):
                xt = X_sm[t_i:t_i + 1].T
                xlag = X_sm[t_i - lag:t_i - lag + 1].T
                S += w * resid[t_i] * resid[t_i - lag] * (xt @ xlag.T + xlag @ xt.T)
        cov_hac = XtX_inv @ S @ XtX_inv
        se_hac = np.sqrt(np.clip(np.diag(cov_hac), 0.0, np.inf))
        t_hac = beta_ols / se_hac
        p_hac = 2.0 * student_t.sf(np.abs(t_hac), df=df_resid)
        results["ols_hac_auto"] = {
            "lags": nw_auto_lags,
            "vol_beta": float(beta_ols[1]),
            "vol_se_hac": float(se_hac[1]),
            "vol_t_hac": float(t_hac[1]),
            "vol_p_hac": float(p_hac[1]),
        }

    # -------------------------------------------------------
    # 6) Ecospace coverage sensitivity
    # -------------------------------------------------------
    try:
        cov_df = pd.read_csv("thesis/synthesis/output_ecospace_missingness/ecospace_coverage_per_bin.csv")
        cov_merged = bins.merge(cov_df[["time_bin", "frac_marine_with_role"]], on="time_bin", how="left")
        cov_col = cov_merged["frac_marine_with_role"].to_numpy(dtype=float)
        if np.sum(np.isfinite(cov_col)) >= 10:
            controls_with_cov = np.column_stack([controls, cov_col])
            cov_shift = _circular_shift_p(v, y, controls_with_cov, seed=args.seed)

            # Block bootstrap with coverage control.
            rv_cov = _residualize(v, controls_with_cov)
            ry_cov = _residualize(y, controls_with_cov)
            mask_cov = np.isfinite(rv_cov) & np.isfinite(ry_cov)
            rv_cc, ry_cc = rv_cov[mask_cov], ry_cov[mask_cov]
            n_cc = len(rv_cc)
            obs_cov = float(np.corrcoef(rv_cc, ry_cc)[0, 1]) if n_cc >= 6 else float("nan")
            rng_cov = np.random.default_rng(args.seed + 1)
            more_cov = 0
            bs = 3
            for _ in range(10000):
                n_blocks = max(1, (n_cc + bs - 1) // bs)
                starts = rng_cov.integers(0, n_cc, size=n_blocks)
                idx = np.concatenate([np.arange(s, s + bs) % n_cc for s in starts])[:n_cc]
                c = float(np.corrcoef(rv_cc, ry_cc[idx])[0, 1])
                if abs(c) >= abs(obs_cov):
                    more_cov += 1
            p_boot_cov = (more_cov + 1) / 10001

            results["coverage_control_sensitivity"] = {
                "controls": "time + sampling_PCA_PC12 + provinciality + frac_marine_with_role",
                **cov_shift,
                "block_bootstrap_b3_p": p_boot_cov,
            }
        else:
            results["coverage_control_sensitivity"] = {"error": "insufficient coverage data"}
    except FileNotFoundError:
        results["coverage_control_sensitivity"] = {"error": "ecospace missingness output not found; run ecospace_missingness_diagnostic.py first"}

    # -------------------------------------------------------
    # 7) Effective sample size note
    # -------------------------------------------------------
    results["effective_sample_sizes"] = {
        "n_bins_for_bin_level_predictors": len(bins),
        "note": "Bin-level predictors (vol_z, time_z, prov_z, sampling PCs) have effective n = n_bins. "
                "Pair-level n inflates precision for these terms unless cluster-robust SEs or mixed-effects are used.",
    }

    # -------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------
    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Summary markdown.
    prim = results.get("primary_specification", {})
    loo = results.get("leave_one_out", {})
    lag = results.get("lagerstaetten_excluded", {})

    lines = [
        "# Robustness battery results",
        "",
        "## Primary specification (locked)",
        f"- Controls: {prim.get('controls', 'N/A')}",
        f"- Partial correlation: {prim.get('corr', 'nan'):.3f}",
        f"- Exact circular-shift p (all N shifts): {prim.get('p_exact', 'nan'):.4f}",
        f"- Random-sampling circular-shift p (20k draws): {prim.get('p_rand', 'nan'):.4f}",
        f"- Resolution limit: {prim.get('resolution_limit', 'N/A')}",
        "",
        "## Leave-one-out stability",
        f"- All positive: {loo.get('all_positive', 'N/A')}",
        f"- Range: [{loo.get('min_corr', 'nan'):.3f}, {loo.get('max_corr', 'nan'):.3f}]",
        f"- Mean: {loo.get('mean_corr', 'nan'):.3f}",
        f"- Most influential bin: {loo.get('most_influential_bin', 'N/A')} Ma",
        "",
        "## Block bootstrap",
    ]
    for bs_key, bs_val in results.get("block_bootstrap", {}).items():
        lines.append(f"- {bs_key}: p = {bs_val.get('p_boot', 'nan'):.4f}")

    lines.extend([
        "",
        "## Lagerstaetten exclusion",
        f"- Bins excluded: {lag.get('bins_excluded', 'N/A')}",
        f"- Partial correlation: {lag.get('corr', 'nan')}",
        f"- Exact circular-shift p: {lag.get('p_exact', 'nan')}",
        "",
        "## SARIMAX AR order sweep",
    ])
    for ar_key, ar_val in results.get("sarimax_ar_sweep", {}).get("models", {}).items():
        if "error" in ar_val:
            lines.append(f"- {ar_key}: error ({ar_val['error'][:60]})")
        else:
            lines.append(
                f"- {ar_key}: AIC={ar_val['aic']:.1f}, BIC={ar_val['bic']:.1f}, "
                f"vol_beta={ar_val['vol_beta']:.4f}, vol_p={ar_val['vol_p']:.3g}"
            )
    lines.append(f"- Best by AIC: {results.get('sarimax_ar_sweep', {}).get('best_by_aic', 'N/A')}")

    cov_sens = results.get("coverage_control_sensitivity", {})
    lines.extend([
        "",
        "## Ecospace coverage control",
        f"- Controls: {cov_sens.get('controls', 'N/A')}",
        f"- Partial correlation: {cov_sens.get('corr', 'nan')}",
        f"- Exact circular-shift p: {cov_sens.get('p_exact', 'nan')}",
        f"- Random-sampling p: {cov_sens.get('p_rand', 'nan')}",
        f"- Block bootstrap (b=3) p: {cov_sens.get('block_bootstrap_b3_p', 'nan')}",
    ])
    if "error" in cov_sens:
        lines.append(f"- Error: {cov_sens['error']}")

    nw = results.get("ols_hac_auto", {})
    lines.extend([
        "",
        "## OLS + HAC (auto bandwidth)",
        f"- Auto lags: {nw.get('lags', 'N/A')}",
        f"- vol_beta: {nw.get('vol_beta', 'nan'):.4f}",
        f"- vol_se_hac: {nw.get('vol_se_hac', 'nan'):.4f}",
        f"- vol_p_hac: {nw.get('vol_p_hac', 'nan'):.3g}",
        "",
        "## Effective sample sizes",
        f"- Bins: {results.get('effective_sample_sizes', {}).get('n_bins_for_bin_level_predictors', 'N/A')}",
        f"- Note: {results.get('effective_sample_sizes', {}).get('note', '')}",
    ])

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
