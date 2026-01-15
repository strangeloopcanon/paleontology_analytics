from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
    sd = float(np.std(x[mask], ddof=0))
    sd = sd if sd > 0 else 1.0
    out[mask] = (x[mask] - mu) / sd
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 4:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yy = y[mask]
    XX = X[mask]
    if len(yy) < 4:
        out = np.full_like(y, fill_value=np.nan, dtype=float)
        return out
    A = np.column_stack([np.ones(len(yy)), XX])
    coef, _, _, _ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A.dot(coef)
    out = np.full_like(y, fill_value=np.nan, dtype=float)
    out[mask] = resid
    return out


@dataclass(frozen=True)
class ShiftCorrResult:
    corr: float
    p_shift_exact: float
    n_bins: int


def _partial_corr_shift_exact(*, x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> ShiftCorrResult:
    x = x.astype(float)
    y = y.astype(float)
    controls = controls.astype(float)
    n = int(len(x))
    if n < 8 or len(y) != n or controls.shape[0] != n:
        return ShiftCorrResult(corr=float("nan"), p_shift_exact=float("nan"), n_bins=n)

    x_res = _ols_residuals(x, controls)
    y_res = _ols_residuals(y, controls)
    obs = _corr(x_res, y_res)
    if not np.isfinite(obs):
        return ShiftCorrResult(corr=float("nan"), p_shift_exact=float("nan"), n_bins=n)

    corrs = []
    for shift in range(n):
        xs = np.roll(x, shift)
        xs_res = _ols_residuals(xs, controls)
        corrs.append(_corr(xs_res, y_res))
    arr = np.asarray(corrs, dtype=float)
    if not np.all(np.isfinite(arr)):
        return ShiftCorrResult(corr=obs, p_shift_exact=float("nan"), n_bins=n)
    p = float(np.mean(np.abs(arr) >= abs(obs)))
    return ShiftCorrResult(corr=obs, p_shift_exact=p, n_bins=n)


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
        Xu = Xg.T.dot(ug)
        S += np.outer(Xu, Xu)

    scale = 1.0
    if g > 1 and (n - p) > 0:
        scale = (g / (g - 1)) * ((n - 1) / (n - p))
    V = scale * (XtX_inv.dot(S).dot(XtX_inv))
    se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf))
    t_stat = beta / se

    df = max(g - 1, 1)
    p_vals = 2.0 * student_t.sf(np.abs(t_stat), df=df)

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "beta": beta,
        "se_cluster": se,
        "t_cluster": t_stat,
        "p_cluster": p_vals,
        "df_cluster": int(df),
        "r2": float(r2),
        "n": int(n),
        "p": int(p),
        "n_clusters": int(g),
        "clusters": uniq,
    }


def _plot_scatter(x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 6:
        return
    xx = x[mask]
    yy = y[mask]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(xx, yy, s=55, alpha=0.85, color="#1f77b4", edgecolors="none")
    A = np.vstack([xx, np.ones(len(xx))]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    xgrid = np.linspace(float(np.min(xx)), float(np.max(xx)), 80)
    ygrid = coef[0] * xgrid + coef[1]
    ax.plot(xgrid, ygrid, color="black", linewidth=1.2, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="thesis/synthesis/output_low_energy_index_mediation_v1")
    p.add_argument(
        "--pair-level-merged",
        default="thesis/synthesis/output_pair_level_model_volatility_v1/merged_pairs.csv",
        help="Pair-level merged file from the publication model (includes z-scored controls).",
    )
    p.add_argument(
        "--diet-frac",
        default="thesis/synthesis/output_role_jobs_volatility_v1/diet_mean_locality_frac_long.csv",
        help="Long table: mean within-locality share per diet category.",
    )
    p.add_argument(
        "--motility-frac",
        default="thesis/synthesis/output_role_jobs_volatility_v1/motility_mean_locality_frac_long.csv",
        help="Long table: mean within-locality share per motility category.",
    )
    p.add_argument(
        "--habit-frac",
        default="thesis/synthesis/output_role_jobs_volatility_v1/habit_mean_locality_frac_long.csv",
        help="Long table: mean within-locality share per life-habit category.",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # Predefined (small) “low-energy / sit-and-filter” vs “high-energy / mobile predator” axis groups.
    # These are intentionally coarse; we test a single composite index to avoid a many-comparisons story.
    LOW_DIET = {"suspension feeder", "deposit feeder", "detritivore"}
    HIGH_DIET = {"carnivore", "piscivore"}
    LOW_MOTILITY = {"stationary", "slow-moving", "passively mobile"}
    HIGH_MOTILITY = {"actively mobile", "fast-moving"}
    LOW_HABIT = {"epifaunal", "infaunal", "semi-infaunal"}
    HIGH_HABIT = {"nektonic", "nektobenthic", "aquatic, depth=surface", "aquatic"}

    df_pairs = pd.read_csv(args.pair_level_merged)
    bin_controls = (
        df_pairs[
            [
                "time_bin",
                "vol_z",
                "time_z",
                "pc1_z",
                "pc2_z",
                "prov_z",
                "delta_from_prev_T_field_meanabs",
                "provinciality",
            ]
        ]
        .drop_duplicates(subset=["time_bin"])
        .sort_values("time_bin", ascending=False)
        .reset_index(drop=True)
    )
    bins = bin_controls["time_bin"].to_numpy(dtype=float)
    vol_z = bin_controls["vol_z"].to_numpy(dtype=float)
    controls = bin_controls[["time_z", "pc1_z", "pc2_z", "prov_z"]].to_numpy(dtype=float)

    def _sum_groups(path: str, *, cat_col: str, low: set[str], high: set[str]) -> pd.DataFrame:
        d = pd.read_csv(path)
        d["time_bin"] = pd.to_numeric(d["time_bin"], errors="coerce")
        d = d.dropna(subset=["time_bin"]).copy()
        d[cat_col] = d[cat_col].astype(str)
        d = d[d["time_bin"].isin(set(bins.tolist()))].copy()
        wide = d.pivot_table(index="time_bin", columns=cat_col, values="mean_locality_frac", fill_value=0.0, aggfunc="mean")
        wide = wide.reindex(bins).fillna(0.0)
        low_cols = [c for c in wide.columns if str(c) in low]
        high_cols = [c for c in wide.columns if str(c) in high]
        out = pd.DataFrame(
            {
                "time_bin": wide.index.to_numpy(dtype=float),
                "low_share": wide[low_cols].sum(axis=1).to_numpy(dtype=float) if low_cols else 0.0,
                "high_share": wide[high_cols].sum(axis=1).to_numpy(dtype=float) if high_cols else 0.0,
            }
        )
        out["balance"] = out["low_share"] - out["high_share"]
        return out

    diet = _sum_groups(args.diet_frac, cat_col="diet_coarse", low=LOW_DIET, high=HIGH_DIET).rename(
        columns={"low_share": "diet_low", "high_share": "diet_high", "balance": "diet_balance"}
    )
    mot = _sum_groups(args.motility_frac, cat_col="motility_coarse", low=LOW_MOTILITY, high=HIGH_MOTILITY).rename(
        columns={"low_share": "mot_low", "high_share": "mot_high", "balance": "mot_balance"}
    )
    hab = _sum_groups(args.habit_frac, cat_col="habit_coarse", low=LOW_HABIT, high=HIGH_HABIT).rename(
        columns={"low_share": "hab_low", "high_share": "hab_high", "balance": "hab_balance"}
    )

    bins_df = bin_controls.merge(diet, on="time_bin", how="left").merge(mot, on="time_bin", how="left").merge(hab, on="time_bin", how="left")
    for c in ["diet_low", "diet_high", "diet_balance", "mot_low", "mot_high", "mot_balance", "hab_low", "hab_high", "hab_balance"]:
        bins_df[c] = pd.to_numeric(bins_df[c], errors="coerce")

    # Composite index: sum of axis balances (low - high) across diet, motility, habit.
    bins_df["low_energy_index_raw"] = bins_df["diet_balance"] + bins_df["mot_balance"] + bins_df["hab_balance"]
    bins_df["low_energy_index_z"] = _z(bins_df["low_energy_index_raw"].to_numpy(dtype=float))
    bins_df.to_csv(out_dir / "bin_index.csv", index=False)

    # Mechanism test: does volatility predict the low-energy index (controlling time+sampling+provinciality)?
    idx_z = bins_df["low_energy_index_z"].to_numpy(dtype=float)
    assoc = _partial_corr_shift_exact(x=vol_z, y=idx_z, controls=controls)
    (out_dir / "assoc_index_vs_volatility.json").write_text(
        json.dumps(
            {
                "partial_corr": assoc.corr,
                "p_shift_exact": assoc.p_shift_exact,
                "n_bins": assoc.n_bins,
                "controls": ["time_z", "sampling_pc1_z", "sampling_pc2_z", "prov_z"],
                "index_definition": {
                    "low_diet": sorted(LOW_DIET),
                    "high_diet": sorted(HIGH_DIET),
                    "low_motility": sorted(LOW_MOTILITY),
                    "high_motility": sorted(HIGH_MOTILITY),
                    "low_habit": sorted(LOW_HABIT),
                    "high_habit": sorted(HIGH_HABIT),
                    "composite": "diet_balance + mot_balance + hab_balance (each = low_share - high_share)",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _plot_scatter(vol_z, idx_z, xlabel="vol_z", ylabel="low_energy_index_z", title="Low-energy index vs volatility", out_path=fig_dir / "index_vs_vol.png")

    # Mediation / attenuation: does adding the low-energy index reduce the volatility coefficient?
    df = df_pairs.merge(bins_df[["time_bin", "low_energy_index_z"]], on="time_bin", how="left")
    df = df.dropna(subset=["functional_similarity_js", "taxsim", "vol_z", "taxsim_x_vol", "time_z", "pc1_z", "pc2_z", "prov_z", "low_energy_index_z"]).copy()

    y = df["functional_similarity_js"].to_numpy(dtype=float)
    taxsim = df["taxsim"].to_numpy(dtype=float)
    volz = df["vol_z"].to_numpy(dtype=float)
    taxsim_x_vol = df["taxsim_x_vol"].to_numpy(dtype=float)
    idxz = df["low_energy_index_z"].to_numpy(dtype=float)
    timez = df["time_z"].to_numpy(dtype=float)
    pc1z = df["pc1_z"].to_numpy(dtype=float)
    pc2z = df["pc2_z"].to_numpy(dtype=float)
    provz = df["prov_z"].to_numpy(dtype=float)
    clusters = df["time_bin"].to_numpy(dtype=float)

    X_no_index = np.column_stack([np.ones(len(df)), taxsim, volz, taxsim_x_vol, timez, pc1z, pc2z, provz])
    X_with_index = np.column_stack([np.ones(len(df)), taxsim, volz, taxsim_x_vol, idxz, timez, pc1z, pc2z, provz])
    terms_no = ["intercept", "taxsim", "vol_z", "taxsim_x_vol_z", "time_z", "sampling_pc1_z", "sampling_pc2_z", "prov_z"]
    terms_with = [
        "intercept",
        "taxsim",
        "vol_z",
        "taxsim_x_vol_z",
        "low_energy_index_z",
        "time_z",
        "sampling_pc1_z",
        "sampling_pc2_z",
        "prov_z",
    ]

    fit_no = _cluster_robust_se(X_no_index, y, clusters=clusters)
    fit_with = _cluster_robust_se(X_with_index, y, clusters=clusters)

    def _coef_table(fit: dict[str, Any], terms: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "term": terms,
                "beta": fit["beta"],
                "se_cluster": fit["se_cluster"],
                "t_cluster": fit["t_cluster"],
                "p_cluster": fit["p_cluster"],
            }
        )

    tab_no = _coef_table(fit_no, terms_no)
    tab_with = _coef_table(fit_with, terms_with)
    tab_no.to_csv(out_dir / "coef_no_index.csv", index=False)
    tab_with.to_csv(out_dir / "coef_with_index.csv", index=False)

    beta_vol_no = float(tab_no.loc[tab_no["term"] == "vol_z", "beta"].iloc[0])
    beta_vol_with = float(tab_with.loc[tab_with["term"] == "vol_z", "beta"].iloc[0])
    attenuation = (beta_vol_no - beta_vol_with) / beta_vol_no if beta_vol_no != 0 else float("nan")

    # Circular-shift nulls for coefficients in the "with index" model.
    # Shift vol_z across bins (preserves vol autocorrelation), refit, compute |t|.
    bins_unique = np.unique(clusters)
    bins_unique = np.sort(bins_unique)[::-1]
    bin_to_ix = {float(b): i for i, b in enumerate(bins_unique.tolist())}
    vol_by_bin = np.array([float(bin_controls.loc[bin_controls["time_bin"] == b, "vol_z"].iloc[0]) for b in bins_unique])
    idx_by_bin = np.array([float(bins_df.loc[bins_df["time_bin"] == b, "low_energy_index_z"].iloc[0]) for b in bins_unique])

    def _refit_with_shift(*, shift_vol: int | None, shift_idx: int | None) -> dict[str, Any]:
        vol_shifted = vol_by_bin if shift_vol is None else np.roll(vol_by_bin, int(shift_vol))
        idx_shifted = idx_by_bin if shift_idx is None else np.roll(idx_by_bin, int(shift_idx))
        vol_map = {float(b): float(vol_shifted[i]) for i, b in enumerate(bins_unique)}
        idx_map = {float(b): float(idx_shifted[i]) for i, b in enumerate(bins_unique)}
        vol_new = np.array([vol_map[float(b)] for b in clusters], dtype=float)
        idx_new = np.array([idx_map[float(b)] for b in clusters], dtype=float)
        X = np.column_stack([np.ones(len(df)), taxsim, vol_new, taxsim * vol_new, idx_new, timez, pc1z, pc2z, provz])
        return _cluster_robust_se(X, y, clusters=clusters)

    obs_fit = fit_with
    obs_t_vol = float(tab_with.loc[tab_with["term"] == "vol_z", "t_cluster"].iloc[0])
    obs_t_idx = float(tab_with.loc[tab_with["term"] == "low_energy_index_z", "t_cluster"].iloc[0])

    t_vol_shifts = []
    t_idx_shifts = []
    for s in range(len(bins_unique)):
        f = _refit_with_shift(shift_vol=s, shift_idx=None)
        # terms: intercept,taxsim,vol,taxsim_x_vol,idx,time,pc1,pc2,prov
        t_vol_shifts.append(float(f["t_cluster"][2]))
        f2 = _refit_with_shift(shift_vol=None, shift_idx=s)
        t_idx_shifts.append(float(f2["t_cluster"][4]))

    p_vol_shift = float(np.mean(np.abs(np.asarray(t_vol_shifts)) >= abs(obs_t_vol)))
    p_idx_shift = float(np.mean(np.abs(np.asarray(t_idx_shifts)) >= abs(obs_t_idx)))

    summary = {
        "pairs_used": int(len(df)),
        "bins_used": int(len(bins_unique)),
        "index_definition": {
            "low_diet": sorted(LOW_DIET),
            "high_diet": sorted(HIGH_DIET),
            "low_motility": sorted(LOW_MOTILITY),
            "high_motility": sorted(HIGH_MOTILITY),
            "low_habit": sorted(LOW_HABIT),
            "high_habit": sorted(HIGH_HABIT),
            "composite": "diet_balance + mot_balance + hab_balance (each = low_share - high_share)",
        },
        "assoc_index_vs_volatility": {"partial_corr": assoc.corr, "p_shift_exact": assoc.p_shift_exact},
        "attenuation_vol_beta": float(attenuation),
        "vol_beta_no_index": beta_vol_no,
        "vol_beta_with_index": beta_vol_with,
        "circular_shift_p_values": {
            "p(vol_z | with_index_model)": p_vol_shift,
            "p(low_energy_index_z | with_index_model)": p_idx_shift,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    summary_md = [
        "# Low-energy / sit-and-filter index (mechanism + mediation)",
        "",
        "This defines a *single preregistered composite index* from ecospace categories and tests:",
        "1) whether it tracks volatility (sampling+autocorr-aware), and",
        "2) whether it attenuates the volatility term in the pair-level convergence model.",
        "",
        f"- bins: {summary['bins_used']}",
        f"- pairs: {summary['pairs_used']}",
        "",
        "## Index definition",
        f"- low diet: {', '.join(sorted(LOW_DIET))}",
        f"- high diet: {', '.join(sorted(HIGH_DIET))}",
        f"- low motility: {', '.join(sorted(LOW_MOTILITY))}",
        f"- high motility: {', '.join(sorted(HIGH_MOTILITY))}",
        f"- low habit: {', '.join(sorted(LOW_HABIT))}",
        f"- high habit: {', '.join(sorted(HIGH_HABIT))}",
        "- index_raw = (diet_low - diet_high) + (mot_low - mot_high) + (hab_low - hab_high); index_z standardized across bins",
        "",
        "## Mechanism test (index vs volatility; controls time+sampling PCA+provinciality)",
        f"- partial corr = {assoc.corr:.3f}",
        f"- circular-shift p (exact) = {assoc.p_shift_exact:.3f}",
        "",
        "## Mediation / attenuation (pair-level model)",
        f"- vol_z beta (no index) = {beta_vol_no:.4f}",
        f"- vol_z beta (with index) = {beta_vol_with:.4f}",
        f"- attenuation = {attenuation:.3f}",
        f"- circular-shift p(vol_z | with index) = {p_vol_shift:.3f}",
        f"- circular-shift p(index | with index) = {p_idx_shift:.3f}",
        "",
        "## Outputs",
        f"- bin index: `{out_dir / 'bin_index.csv'}`",
        f"- coefficients (no index): `{out_dir / 'coef_no_index.csv'}`",
        f"- coefficients (with index): `{out_dir / 'coef_with_index.csv'}`",
        f"- summary JSON: `{out_dir / 'summary.json'}`",
        f"- figure: `{fig_dir / 'index_vs_vol.png'}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_md) + "\n")


if __name__ == "__main__":
    main()

