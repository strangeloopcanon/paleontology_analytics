from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import t as student_t
from scipy.stats import norm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _z(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if int(np.sum(mask)) < 3:
        return out
    mu = float(np.mean(x[mask]))
    sd = float(np.std(x[mask], ddof=0))
    sd = sd if sd > 0 else 1.0
    out[mask] = (x[mask] - mu) / sd
    return out


def _pca_scores(X: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(float)
    mask = np.all(np.isfinite(X), axis=1)
    if int(np.sum(mask)) < max(6, k + 3):
        return np.full((len(X), k), np.nan), np.full(k, np.nan), np.full((k, X.shape[1]), np.nan)
    Xc = X[mask]
    mu = np.mean(Xc, axis=0)
    sd = np.std(Xc, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Xc - mu) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S**2) / np.sum(S**2)
    kk = min(int(k), Vt.shape[0])
    scores = np.full((len(X), k), np.nan, dtype=float)
    scores[mask, :kk] = U[:, :kk] * S[:kk]
    explained = np.full(k, np.nan, dtype=float)
    explained[:kk] = var[:kk]
    loadings = np.full((k, X.shape[1]), np.nan, dtype=float)
    loadings[:kk, :] = Vt[:kk, :]
    return scores, explained, loadings


def _coef_table(res: Any, *, term_names: list[str]) -> pd.DataFrame:
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    t = params / bse
    out = pd.DataFrame({"term": term_names, "beta": params, "se": bse, "t": t})
    if hasattr(res, "pvalues"):
        out["p"] = np.asarray(res.pvalues, dtype=float)
    return out


def _plot_fitted(y: np.ndarray, yhat: np.ndarray, *, title: str, out_path: Path) -> None:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if int(np.sum(mask)) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(y[mask], yhat[mask], s=55, alpha=0.85, color="#1f77b4", edgecolors="none")
    lo = float(min(np.min(y[mask]), np.min(yhat[mask])))
    hi = float(max(np.max(y[mask]), np.max(yhat[mask])))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("observed")
    ax.set_ylabel("fitted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="thesis/synthesis/output_time_series_hierarchical_models_v1")
    p.add_argument(
        "--bins",
        default="thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv",
        help="Bin-level merged table (includes Macrostrat proxies).",
    )
    p.add_argument(
        "--pairs",
        default="thesis/synthesis/output_pair_level_model_volatility_v1/merged_pairs.csv",
        help="Pair-level merged table from publication model (taxsim, vol_z, controls).",
    )
    p.add_argument("--ar-order", type=int, default=1, help="AR order for SARIMAX residuals.")
    p.add_argument("--hac-maxlags", type=int, default=1, help="Max lags for Newey-West HAC SEs on OLS.")
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # -------------------------
    # A) Bin-level time-series model (AR errors / state-space)
    # -------------------------
    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    # Sampling PCA (same features as the publication pair-level model).
    feat_names = [
        "log1p(n_localities)",
        "log1p(marine_n_collections)",
        "log1p(marine_n_occurrences)",
        "log1p(macro_col_area_sum)",
        "log1p(macro_n_sections)",
    ]
    feats = np.column_stack(
        [
            np.log1p(bins["n_localities"].to_numpy(dtype=float)),
            np.log1p(bins["marine_n_collections"].to_numpy(dtype=float)),
            np.log1p(bins["marine_n_occurrences"].to_numpy(dtype=float)),
            np.log1p(bins["macro_col_area_sum"].to_numpy(dtype=float)),
            np.log1p(bins["macro_n_sections"].to_numpy(dtype=float)),
        ]
    )
    pcs, pc_expl, pc_load = _pca_scores(feats, k=2)
    bins["sampling_pc1"] = pcs[:, 0]
    bins["sampling_pc2"] = pcs[:, 1]
    (out_dir / "sampling_pca.json").write_text(
        json.dumps(
            {"feature_names": feat_names, "explained_variance": pc_expl.tolist(), "loadings": pc_load.tolist()}, indent=2
        )
        + "\n"
    )

    bins["y"] = pd.to_numeric(bins["functional_excess_similarity_js"], errors="coerce")
    bins["vol_z"] = _z(pd.to_numeric(bins["delta_from_prev_T_field_meanabs"], errors="coerce").to_numpy(dtype=float))
    bins["time_z"] = _z(pd.to_numeric(bins["time_bin"], errors="coerce").to_numpy(dtype=float))
    bins["prov_z"] = _z(pd.to_numeric(bins["provinciality"], errors="coerce").to_numpy(dtype=float))
    bins["pc1_z"] = _z(pd.to_numeric(bins["sampling_pc1"], errors="coerce").to_numpy(dtype=float))
    bins["pc2_z"] = _z(pd.to_numeric(bins["sampling_pc2"], errors="coerce").to_numpy(dtype=float))

    bins = bins.dropna(subset=["y", "vol_z", "time_z", "prov_z", "pc1_z", "pc2_z"]).copy()
    bins.to_csv(out_dir / "bins_model_table.csv", index=False)

    y = bins["y"].to_numpy(dtype=float)
    X = bins[["vol_z", "time_z", "pc1_z", "pc2_z", "prov_z"]].to_numpy(dtype=float)
    X_sm = np.column_stack([np.ones(len(y)), X])
    term_names = ["intercept", "vol_z", "time_z", "sampling_pc1_z", "sampling_pc2_z", "prov_z"]

    # OLS with iid SEs
    beta_ols, *_ = np.linalg.lstsq(X_sm, y, rcond=None)
    resid = y - X_sm.dot(beta_ols)
    n = int(len(y))
    p = int(X_sm.shape[1])
    df_resid = max(n - p, 1)
    sigma2 = float(np.sum(resid**2) / df_resid)
    XtX_inv = np.linalg.inv(X_sm.T.dot(X_sm))
    cov_iid = sigma2 * XtX_inv
    se_iid = np.sqrt(np.clip(np.diag(cov_iid), 0.0, np.inf))
    t_iid = beta_ols / se_iid
    p_iid = 2.0 * student_t.sf(np.abs(t_iid), df=df_resid)
    ols_tab = pd.DataFrame(
        {"term": term_names, "beta": beta_ols, "se": se_iid, "t": t_iid, "p": p_iid, "model": "OLS"}
    )

    # OLS + Newey-West HAC covariance (Bartlett kernel).
    L = int(args.hac_maxlags)
    S = np.zeros((p, p), dtype=float)
    # lag 0
    for t in range(n):
        xt = X_sm[t : t + 1].T
        S += (resid[t] ** 2) * (xt @ xt.T)
    # lags 1..L
    for lag in range(1, L + 1):
        w = 1.0 - lag / float(L + 1)
        for t in range(lag, n):
            xt = X_sm[t : t + 1].T
            xlag = X_sm[t - lag : t - lag + 1].T
            S += w * resid[t] * resid[t - lag] * (xt @ xlag.T + xlag @ xt.T)
    cov_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.clip(np.diag(cov_hac), 0.0, np.inf))
    t_hac = beta_ols / se_hac
    p_hac = 2.0 * student_t.sf(np.abs(t_hac), df=df_resid)
    ols_hac_tab = pd.DataFrame(
        {
            "term": term_names,
            "beta": beta_ols,
            "se": se_hac,
            "t": t_hac,
            "p": p_hac,
            "model": f"OLS_HAC_lags{int(args.hac_maxlags)}",
        }
    )

    # SARIMAX with AR errors (state-space): y_t = X_t beta + e_t; e_t AR(p).
    ar_p = int(args.ar_order)
    sar = SARIMAX(
        endog=y,
        exog=X,
        order=(ar_p, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    # SARIMAX params include intercept + exog betas + AR params + sigma2.
    sar_params = np.asarray(sar.params, dtype=float)
    sar_bse = np.asarray(sar.bse, dtype=float)
    sar_terms = term_names + [f"ar_L{i+1}" for i in range(ar_p)] + ["sigma2"]
    if len(sar_terms) != len(sar_params):
        # Fall back to statsmodels-provided names if the expected layout changes.
        sar_terms = list(getattr(sar, "param_names", [f"param_{i}" for i in range(len(sar_params))]))
    sar_t = sar_params / sar_bse
    sar_tab = pd.DataFrame(
        {
            "term": sar_terms,
            "beta": sar_params,
            "se": sar_bse,
            "t": sar_t,
            "p": np.asarray(sar.pvalues, dtype=float),
            "model": f"SARIMAX_AR{ar_p}",
        }
    )

    bin_coef = pd.concat([ols_tab, ols_hac_tab, sar_tab], ignore_index=True)
    bin_coef.to_csv(out_dir / "bin_model_coefs.csv", index=False)

    # Fitted plots
    _plot_fitted(y, X_sm.dot(beta_ols), title="Bin model: OLS fitted", out_path=fig_dir / "bins_ols_fitted.png")
    _plot_fitted(
        y,
        np.asarray(sar.fittedvalues, dtype=float),
        title=f"Bin model: SARIMAX AR({ar_p}) fitted",
        out_path=fig_dir / "bins_sarimax_fitted.png",
    )

    # -------------------------
    # B) Pair-level hierarchical model (mixed effects by time bin)
    # -------------------------
    pairs = pd.read_csv(args.pairs)
    pairs = pairs.dropna(
        subset=["functional_similarity_js", "taxsim", "vol_z", "taxsim_x_vol", "time_z", "pc1_z", "pc2_z", "prov_z", "time_bin"]
    ).copy()
    pairs["time_bin"] = pd.to_numeric(pairs["time_bin"], errors="coerce")
    pairs = pairs.dropna(subset=["time_bin"]).copy()
    pairs["group"] = pairs["time_bin"].astype(float)

    # Random intercept + random slope for taxsim by bin (captures varying functional↔taxonomic coupling).
    y_p = pairs["functional_similarity_js"].to_numpy(dtype=float)
    exog_core = pairs[["taxsim", "vol_z", "taxsim_x_vol", "time_z", "pc1_z", "pc2_z", "prov_z"]].to_numpy(dtype=float)
    exog = np.column_stack([np.ones(len(pairs)), exog_core])
    exog_names = ["intercept", "taxsim", "vol_z", "taxsim_x_vol_z", "time_z", "sampling_pc1_z", "sampling_pc2_z", "prov_z"]
    exog_re = np.column_stack([np.ones(len(pairs)), pairs["taxsim"].to_numpy(dtype=float)])  # random intercept + random slope

    md = MixedLM(endog=y_p, exog=exog, groups=pairs["group"], exog_re=exog_re)
    try:
        mdf = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        mixed_ok = True
    except Exception as e:
        # Fallback: random intercept only.
        mixed_ok = False
        mdf = None
        err = str(e)
        md2 = MixedLM(endog=y_p, exog=exog, groups=pairs["group"])
        mdf2 = md2.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        mdf = mdf2

    fe = np.asarray(getattr(mdf, "fe_params", mdf.params[: len(exog_names)]), dtype=float)
    fe_se = np.asarray(getattr(mdf, "bse_fe", mdf.bse[: len(exog_names)]), dtype=float)
    fe_t = fe / fe_se
    fe_p = 2.0 * norm.sf(np.abs(fe_t))
    mixed_tab = pd.DataFrame({"term": exog_names, "beta": fe, "se": fe_se, "t": fe_t, "p": fe_p})
    mixed_tab["model"] = "MixedLM_re_int+taxsim" if mixed_ok else "MixedLM_re_int_only"
    mixed_tab.to_csv(out_dir / "pair_mixedlm_coefs.csv", index=False)

    # Save fit metadata
    meta = {
        "bins_used": int(pairs["group"].nunique()),
        "pairs_used": int(len(pairs)),
        "mixedlm_random_slope_taxsim": bool(mixed_ok),
        "mixedlm_fallback_error": None if mixed_ok else err,
        "sarimax_ar_order": ar_p,
        "hac_maxlags": int(args.hac_maxlags),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # Markdown summary (compact).
    def _pick(df: pd.DataFrame, model: str, term: str) -> dict[str, float]:
        r = df[(df["model"] == model) & (df["term"] == term)]
        if len(r) != 1:
            return {"beta": float("nan"), "se": float("nan"), "p": float("nan")}
        row = r.iloc[0]
        return {"beta": float(row["beta"]), "se": float(row["se"]), "p": float(row.get("p", float("nan")))}

    ols_vol = _pick(bin_coef, "OLS", "vol_z")
    hac_vol = _pick(bin_coef, f"OLS_HAC_lags{int(args.hac_maxlags)}", "vol_z")
    sar_vol = _pick(bin_coef, f"SARIMAX_AR{ar_p}", "vol_z")
    mixed_vol = _pick(mixed_tab, mixed_tab["model"].iloc[0], "vol_z")

    summary_md = [
        "# Time-series + hierarchical inference (publication-grade upgrade)",
        "",
        "We fit two complementary models:",
        "1) **Bin-level time-series regression** on `functional_excess_similarity_js` with AR errors (SARIMAX) and HAC SE sensitivity.",
        "2) **Pair-level mixed effects** model with random bin effects (and, when possible, a random taxsim slope).",
        "",
        "## Bin-level (n bins)",
        f"- bins: {len(bins)}",
        "",
        "Volatility coefficient (`vol_z`) across bin-level models:",
        f"- OLS: beta={ols_vol['beta']:.4f}, se={ols_vol['se']:.4f}, p={ols_vol['p']:.3g}",
        f"- OLS + HAC: beta={hac_vol['beta']:.4f}, se={hac_vol['se']:.4f}, p={hac_vol['p']:.3g}",
        f"- SARIMAX AR({ar_p}): beta={sar_vol['beta']:.4f}, se={sar_vol['se']:.4f}, p={sar_vol['p']:.3g}",
        "",
        "## Pair-level mixed effects",
        f"- bins: {meta['bins_used']}, pairs: {meta['pairs_used']}",
        f"- model: {mixed_tab['model'].iloc[0]}",
        f"- vol_z: beta={mixed_vol['beta']:.4f}, se={mixed_vol['se']:.4f}, p={mixed_vol['p']:.3g}",
        "",
        "## Outputs",
        f"- bin table: `{out_dir / 'bins_model_table.csv'}`",
        f"- bin coefficients: `{out_dir / 'bin_model_coefs.csv'}`",
        f"- pair mixedLM coefficients: `{out_dir / 'pair_mixedlm_coefs.csv'}`",
        f"- meta: `{out_dir / 'meta.json'}`",
        f"- figures: `{fig_dir}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_md) + "\n")


if __name__ == "__main__":
    main()
