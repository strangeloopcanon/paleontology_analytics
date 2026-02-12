from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import t as student_t


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _z(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if int(np.sum(mask)) < 3:
        return out
    mu = float(np.mean(x[mask]))
    sd = float(np.std(x[mask], ddof=1))
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


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yy = y[mask]
    XX = X[mask]
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX.dot(beta)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return beta, resid, r2


def _cluster_robust_se(X: np.ndarray, y: np.ndarray, *, clusters: np.ndarray) -> dict[str, Any]:
    """
    OLS + CR1 (Arellano) cluster-robust SEs.
    Returns beta, se, t, p, r2, n, p_params, n_clusters.
    """
    y = y.astype(float)
    X = X.astype(float)
    clusters = clusters.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(clusters)
    yy = y[mask]
    XX = X[mask]
    cc = clusters[mask]

    n = int(len(yy))
    p = int(XX.shape[1])
    uniq = np.unique(cc)
    g = int(len(uniq))
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX.dot(beta)

    XtX_inv = np.linalg.inv(XX.T.dot(XX))
    S = np.zeros((p, p), dtype=float)
    for u in uniq:
        idx = cc == u
        Xg = XX[idx]
        ug = resid[idx]
        Xu = Xg.T.dot(ug)  # (p,)
        S += np.outer(Xu, Xu)

    # CR1 small-sample correction.
    scale = 1.0
    if g > 1 and (n - p) > 0:
        scale = (g / (g - 1)) * ((n - 1) / (n - p))
    V = scale * (XtX_inv.dot(S).dot(XtX_inv))
    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf))
    t_stat = beta / se

    df = max(g - 1, 1)
    p_vals = 2.0 * student_t.sf(np.abs(t_stat), df=df)

    # R²
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "beta": beta.tolist(),
        "se_cluster": se.tolist(),
        "t_cluster": t_stat.tolist(),
        "p_cluster": p_vals.tolist(),
        "df_cluster": int(df),
        "r2": float(r2),
        "n": int(n),
        "p": int(p),
        "n_clusters": int(g),
    }


def _plot_bin_points(df_bins: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df_bins[[x, y]].dropna().copy()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(d[x], d[y], s=55, alpha=0.8, color="#1f77b4", edgecolors="none")
    A = np.vstack([d[x].to_numpy(), np.ones(len(d))]).T
    coef, *_ = np.linalg.lstsq(A, d[y].to_numpy(), rcond=None)
    xx = np.linspace(float(d[x].min()), float(d[x].max()), 60)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="black", linewidth=1.2, alpha=0.75)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="thesis/convergence/output_v3_fullpbdb/pairwise_sample.csv")
    p.add_argument("--bins", default="thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv")
    p.add_argument("--out", default="thesis/synthesis/output_pair_level_model_volatility_v1")
    p.add_argument(
        "--permutations",
        type=int,
        default=20000,
        help="Monte Carlo shifts (with replacement). Exact (all shifts) p-values are also computed.",
    )
    p.add_argument("--seed", type=int, default=77)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    pairs = pd.read_csv(args.pairs)
    bins = pd.read_csv(args.bins)

    # Restrict to bins that survive the current end-to-end robustness table.
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)
    keep_bins = set(bins["time_bin"].astype(float).tolist())
    pairs = pairs[pairs["time_bin"].astype(float).isin(keep_bins)].copy()

    # Sampling PCA on the same bin set (handles collinearity between PBDB sampling proxies and Macrostrat).
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

    # Standardize bin-level predictors.
    bins["vol_z"] = _z(bins["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float))
    bins["time_z"] = _z(bins["time_bin"].to_numpy(dtype=float))
    bins["prov_z"] = _z(bins["provinciality"].to_numpy(dtype=float))
    bins["pc1_z"] = _z(bins["sampling_pc1"].to_numpy(dtype=float))
    bins["pc2_z"] = _z(bins["sampling_pc2"].to_numpy(dtype=float))

    # Merge bin-level predictors onto pairwise data.
    df = pairs.merge(
        bins[
            [
                "time_bin",
                "vol_z",
                "time_z",
                "prov_z",
                "pc1_z",
                "pc2_z",
                "delta_from_prev_T_field_meanabs",
                "provinciality",
            ]
        ],
        on="time_bin",
        how="left",
    )
    df = df.dropna(
        subset=[
            "functional_similarity_js",
            "taxonomic_similarity",
            "vol_z",
            "time_z",
            "pc1_z",
            "pc2_z",
            "prov_z",
        ]
    ).copy()
    df["taxsim"] = pd.to_numeric(df["taxonomic_similarity"], errors="coerce")
    df = df.dropna(subset=["taxsim"]).copy()
    df["taxsim_x_vol"] = df["taxsim"] * df["vol_z"]
    df.to_csv(out_dir / "merged_pairs.csv", index=False)

    y = df["functional_similarity_js"].to_numpy(dtype=float)
    taxsim = df["taxsim"].to_numpy(dtype=float)
    vol_z = df["vol_z"].to_numpy(dtype=float)
    taxsim_x_vol = df["taxsim_x_vol"].to_numpy(dtype=float)
    time_z = df["time_z"].to_numpy(dtype=float)
    pc1_z = df["pc1_z"].to_numpy(dtype=float)
    pc2_z = df["pc2_z"].to_numpy(dtype=float)
    prov_z = df["prov_z"].to_numpy(dtype=float)
    clusters = df["time_bin"].to_numpy(dtype=float)

    # Model specs.
    # Interpretability: with taxsim in [0,1], the main vol_z coefficient is an intercept shift at taxsim=0.
    X_base = np.column_stack([np.ones(len(df)), taxsim, time_z, pc1_z, pc2_z, prov_z])
    X_vol = np.column_stack([np.ones(len(df)), taxsim, vol_z, taxsim_x_vol, time_z, pc1_z, pc2_z, prov_z])
    X_vol_no_int = np.column_stack([np.ones(len(df)), taxsim, vol_z, time_z, pc1_z, pc2_z, prov_z])

    base_fit = _cluster_robust_se(X_base, y, clusters=clusters)
    vol_fit = _cluster_robust_se(X_vol, y, clusters=clusters)
    vol_fit_no_int = _cluster_robust_se(X_vol_no_int, y, clusters=clusters)

    # Circular-shift null for volatility effects (bin-level shifts preserve vol autocorrelation).
    # We shift vol_z across bins, recompute the pair-level vol columns, and refit the volatility model.
    rng = np.random.default_rng(int(args.seed))
    bin_order = bins["time_bin"].to_numpy(dtype=float)  # already time-sorted (older->younger)
    vol_by_bin = bins.set_index("time_bin")["vol_z"].to_dict()
    ordered_vol = np.array([float(vol_by_bin[float(tb)]) for tb in bin_order], dtype=float)
    tb_to_index = {float(tb): i for i, tb in enumerate(bin_order)}
    row_bin_idx = np.array([tb_to_index[float(tb)] for tb in clusters], dtype=int)

    def _fit_with_shift(shift: int) -> tuple[float, float]:
        vv = np.roll(ordered_vol, int(shift))
        vol_shift = vv[row_bin_idx]
        Xs = np.column_stack([np.ones(len(df)), taxsim, vol_shift, taxsim * vol_shift, time_z, pc1_z, pc2_z, prov_z])
        beta, _, _ = _ols(Xs, y)
        return float(beta[2]), float(beta[3])  # vol main, interaction

    obs_beta, _, _ = _ols(X_vol, y)
    obs_main = float(obs_beta[2])
    obs_int = float(obs_beta[3])

    n_bins = int(len(bin_order))

    # Exact circular-shift p-values (all distinct shifts, including 0 = observed).
    all_shifts = list(range(0, n_bins))
    more_main_exact = 0
    more_int_exact = 0
    for s in all_shifts:
        b_main, b_int = _fit_with_shift(int(s))
        if abs(b_main) >= abs(obs_main):
            more_main_exact += 1
        if abs(b_int) >= abs(obs_int):
            more_int_exact += 1
    p_main_exact = more_main_exact / float(len(all_shifts))
    p_int_exact = more_int_exact / float(len(all_shifts))

    # Monte Carlo shifts (with replacement) for a smoother estimate (note: with small n_bins,
    # the exact p-values above are the more principled reference).
    shifts = rng.integers(0, n_bins, size=int(args.permutations))
    more_main_mc = 0
    more_int_mc = 0
    for s in shifts:
        b_main, b_int = _fit_with_shift(int(s))
        if abs(b_main) >= abs(obs_main):
            more_main_mc += 1
        if abs(b_int) >= abs(obs_int):
            more_int_mc += 1
    p_main_mc = more_main_mc / float(int(args.permutations))
    p_int_mc = more_int_mc / float(int(args.permutations))

    results = {
        "n_pairs": int(len(df)),
        "n_bins": int(df["time_bin"].nunique()),
        "sampling_pca_pc1_explained": float(pc_expl[0]),
        "model_base": base_fit,
        "model_vol": vol_fit,
        "model_vol_no_interaction": vol_fit_no_int,
        "circular_shift_p": {
            "permutations": int(args.permutations),
            "obs_beta_vol_main": obs_main,
            "obs_beta_vol_interaction": obs_int,
            "p_vol_main_exact": float(p_main_exact),
            "p_vol_interaction_exact": float(p_int_exact),
            "p_vol_main_mc": float(p_main_mc),
            "p_vol_interaction_mc": float(p_int_mc),
        },
    }
    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Bin-level plot: mean residual in the original convergence table vs volatility (context).
    _plot_bin_points(
        bins,
        x="delta_from_prev_T_field_meanabs",
        y="functional_excess_similarity_js",
        out_path=fig_dir / "bins_volatility_vs_excess_similarity.png",
        title="Bins: volatility vs functional excess similarity",
    )

    coef_rows = []
    def _coef_table(fit: dict[str, Any], names: list[str], model: str) -> None:
        for i, nm in enumerate(names):
            coef_rows.append(
                {
                    "model": model,
                    "term": nm,
                    "beta": float(fit["beta"][i]),
                    "se_cluster": float(fit["se_cluster"][i]),
                    "t_cluster": float(fit["t_cluster"][i]),
                    "p_cluster": float(fit["p_cluster"][i]),
                }
            )

    _coef_table(base_fit, ["intercept", "taxsim", "time_z", "sampling_pc1_z", "sampling_pc2_z", "prov_z"], "base")
    _coef_table(
        vol_fit,
        [
            "intercept",
            "taxsim",
            "vol_z",
            "taxsim_x_vol_z",
            "time_z",
            "sampling_pc1_z",
            "sampling_pc2_z",
            "prov_z",
        ],
        "vol+interaction",
    )
    _coef_table(
        vol_fit_no_int,
        ["intercept", "taxsim", "vol_z", "time_z", "sampling_pc1_z", "sampling_pc2_z", "prov_z"],
        "vol_only",
    )
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(out_dir / "coef_table.csv", index=False)

    lines = [
        "# Pair-level model: does volatility shift functional similarity beyond taxonomy?",
        "",
        "This is a publication-oriented upgrade of the bin-level residual correlation by fitting a **pair-level** regression and using",
        "(i) bin-clustered robust SEs and (ii) a time-series circular-shift null on volatility.",
        "",
        "Outcome: `functional_similarity_js` (pairwise JS similarity on role-frequency vectors).",
        "Key predictors:",
        "- `taxsim` = taxonomic Jaccard similarity between locality genera sets (pair-level).",
        "- `vol_z` = standardized climate volatility (bin-level; mean |ΔT| field).",
        "- `taxsim_x_vol_z` tests whether volatility changes the *slope* (coupling), not just the intercept.",
        "",
        "Controls (bin-level): `time_z`, `sampling_pc1_z`, `sampling_pc2_z`, `prov_z`.",
        "",
        f"- pairs used: {results['n_pairs']}",
        f"- bins used: {results['n_bins']}",
        f"- sampling PCA PC1 explained variance: {results['sampling_pca_pc1_explained']:.3f}",
        "",
        "## Cluster-robust inference (clusters = time bins)",
        "",
        f"- base model R²: {base_fit['r2']:.3f}",
        f"- vol+interaction model R²: {vol_fit['r2']:.3f}",
        f"- vol-only model R²: {vol_fit_no_int['r2']:.3f}",
        "",
        "Key terms (vol+interaction):",
        "- `vol_z` (intercept shift at taxsim=0): "
        + "beta={:.4f}, p_cluster={:.3g}".format(
            float(coef_df[(coef_df['model']=='vol+interaction') & (coef_df['term']=='vol_z')]['beta'].iloc[0]),
            float(coef_df[(coef_df['model']=='vol+interaction') & (coef_df['term']=='vol_z')]['p_cluster'].iloc[0]),
        ),
        "- `taxsim_x_vol_z` (slope change): "
        + "beta={:.4f}, p_cluster={:.3g}".format(
            float(coef_df[(coef_df['model']=='vol+interaction') & (coef_df['term']=='taxsim_x_vol_z')]['beta'].iloc[0]),
            float(coef_df[(coef_df['model']=='vol+interaction') & (coef_df['term']=='taxsim_x_vol_z')]['p_cluster'].iloc[0]),
        ),
        "",
        "## Circular-shift null (time-series-aware; volatility shifted across bins)",
        "",
        "- p(vol_z) exact: {p:.3g}".format(p=float(results["circular_shift_p"]["p_vol_main_exact"])),
        "- p(taxsim_x_vol_z) exact: {p:.3g}".format(p=float(results["circular_shift_p"]["p_vol_interaction_exact"])),
        "- p(vol_z) MC: {p:.3g}".format(p=float(results["circular_shift_p"]["p_vol_main_mc"])),
        "- p(taxsim_x_vol_z) MC: {p:.3g}".format(p=float(results["circular_shift_p"]["p_vol_interaction_mc"])),
        "",
        "## Outputs",
        "",
        f"- merged pairs: `{out_dir / 'merged_pairs.csv'}`",
        f"- coefficient table: `{out_dir / 'coef_table.csv'}`",
        f"- stats: `{out_dir / 'analysis_results.json'}`",
        f"- sampling PCA: `{out_dir / 'sampling_pca.json'}`",
        f"- figures: `{fig_dir}`",
        "",
        "Notes:",
        "- This uses the stored pairwise sample from the convergence pipeline (not all possible pairs).",
        "- If volatility acts mainly as a baseline shift (more similar jobs even when taxa differ), expect `vol_z > 0` and `taxsim_x_vol_z ≈ 0`.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
