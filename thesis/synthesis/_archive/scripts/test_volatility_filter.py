from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _z(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 0:
        return x * np.nan
    return (x - mu) / sd


def _ols_beta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    # Returns beta for y ~ X (X already includes intercept if desired).
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yy = y[mask]
    XX = X[mask]
    if len(yy) < (XX.shape[1] + 2):
        return np.full(XX.shape[1], fill_value=np.nan, dtype=float)
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    return beta.astype(float)


def _residualize_multi(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yy = y[mask]
    XX = X[mask]
    out = np.full_like(y, fill_value=np.nan, dtype=float)
    A = np.column_stack([np.ones(len(XX)), XX])
    if len(yy) < (A.shape[1] + 2):
        return out
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A.dot(beta)
    out[mask] = resid
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _perm_test_corr(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 6:
        return {"corr": float("nan"), "p_perm": float("nan"), "n": float(len(x))}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(int(seed))
    more_extreme = 0
    for _ in range(int(permutations)):
        yp = rng.permutation(y)
        c = float(np.corrcoef(x, yp)[0, 1])
        if abs(c) >= abs(obs):
            more_extreme += 1
    p = (more_extreme + 1) / (int(permutations) + 1)
    return {"corr": obs, "p_perm": float(p), "n": float(len(x))}


def _partial_corr_perm(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    # Partial corr via residualization; permute residualized y.
    rx = _residualize_multi(x, controls)
    ry = _residualize_multi(y, controls)
    return _perm_test_corr(rx, ry, permutations=int(permutations), seed=int(seed))


def _bootstrap_mediation(
    v: np.ndarray,
    g: np.ndarray,
    c: np.ndarray,
    controls: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    # Standardized mediation:
    # g ~ v + controls  (a)
    # c ~ v + g + controls (b, direct)
    # indirect = a*b
    rng = np.random.default_rng(int(seed))

    def _fit(vv: np.ndarray, gg: np.ndarray, cc: np.ndarray, XX: np.ndarray) -> tuple[float, float, float]:
        # Standardize within sample.
        vv = _z(vv)
        gg = _z(gg)
        cc = _z(cc)

        # Controls: include intercept and standardized controls.
        Xc = np.column_stack([np.ones(len(vv)), _z(XX[:, 0]), _z(XX[:, 1])])
        # g model
        Xg = np.column_stack([Xc, vv])
        beta_g = _ols_beta(gg, Xg)
        a = float(beta_g[-1])
        # c model
        Xc2 = np.column_stack([Xc, vv, gg])
        beta_c = _ols_beta(cc, Xc2)
        b = float(beta_c[-1])
        direct = float(beta_c[-2])
        return a, b, direct

    # Fit on full sample.
    mask = np.isfinite(v) & np.isfinite(g) & np.isfinite(c) & np.all(np.isfinite(controls), axis=1)
    vv = v[mask]
    gg = g[mask]
    cc = c[mask]
    XX = controls[mask]
    if len(vv) < 10:
        return {"n": int(len(vv)), "a": float("nan"), "b": float("nan"), "direct": float("nan"), "indirect": float("nan")}

    a0, b0, direct0 = _fit(vv, gg, cc, XX)
    indirect0 = a0 * b0

    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(vv), size=len(vv))
        a, b, direct = _fit(vv[idx], gg[idx], cc[idx], XX[idx])
        boots.append((a, b, direct, a * b))
    boots = np.array(boots, dtype=float)

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        lo = float(np.nanquantile(arr, 0.025))
        hi = float(np.nanquantile(arr, 0.975))
        return lo, hi

    return {
        "n": int(len(vv)),
        "a": float(a0),
        "b": float(b0),
        "direct": float(direct0),
        "indirect": float(indirect0),
        "ci": {
            "a": _ci(boots[:, 0]),
            "b": _ci(boots[:, 1]),
            "direct": _ci(boots[:, 2]),
            "indirect": _ci(boots[:, 3]),
        },
    }


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y]].dropna()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.scatter(d[x], d[y], alpha=0.75, s=35, color="#1f77b4", edgecolors="none")
    A = np.vstack([d[x].to_numpy(), np.ones(len(d))]).T
    coef, *_ = np.linalg.lstsq(A, d[y].to_numpy(), rcond=None)
    xx = np.linspace(float(d[x].min()), float(d[x].max()), 60)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="black", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_timeseries(df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    d = df.sort_values("time_bin", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.plot(d["time_bin"], d["volatility"], marker="o", linewidth=1.4, color="#d62728", label="volatility (ΔT field)")
    ax.set_xlabel("Time bin (Ma; older → younger)")
    ax.set_ylabel("Volatility proxy")
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.plot(
        d["time_bin"],
        d["convergence"],
        marker="s",
        linewidth=1.4,
        color="#2ca02c",
        label="convergence (excess role JS)",
    )
    ax2.plot(
        d["time_bin"],
        d["filter_index"],
        marker="^",
        linewidth=1.4,
        color="#1f77b4",
        alpha=0.85,
        label="filter index",
    )
    ax2.set_ylabel("Convergence / filter index")
    ax.set_title(title)

    # Combined legend.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _pivot_occupancy(occ: pd.DataFrame, *, cat_col: str) -> pd.DataFrame:
    d = occ.copy()
    d = d.dropna(subset=["time_bin", cat_col, "occupancy_frac"]).copy()
    p = d.pivot_table(index="time_bin", columns=cat_col, values="occupancy_frac", aggfunc="mean")
    p = p.reset_index().rename_axis(None, axis=1)
    return p


def _safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[existing].mean(axis=1)


def _heterogeneity_from_pivot(pivot: pd.DataFrame, *, prevalence_threshold: float) -> pd.Series:
    # pivot: index=time_bin, columns=categories, values=occupancy_frac (0..1)
    # Heterogeneity proxy: mean p(1-p) over categories with sufficient prevalence.
    p = pivot.copy()
    p = p.fillna(0.0)
    prev = p.mean(axis=0)
    keep = list(prev[prev >= float(prevalence_threshold)].index)
    if not keep:
        return pd.Series([np.nan] * len(p), index=p.index)
    pp = p[keep]
    return (pp * (1.0 - pp)).mean(axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conv-v2", default="thesis/convergence/output_v2/timebin_metrics.csv")
    p.add_argument("--decomp", default="thesis/convergence/output_role_decomposition/timebin_metrics_decomposition.csv")
    p.add_argument("--diet-occ", default="thesis/convergence/output_role_decomposition/diet_occupancy_timeseries.csv")
    p.add_argument("--motility-occ", default="thesis/convergence/output_role_decomposition/motility_occupancy_timeseries.csv")
    p.add_argument("--habit-occ", default="thesis/convergence/output_role_decomposition/habit_occupancy_timeseries.csv")
    p.add_argument("--dino", default="thesis/body_size_stability/output_independent_stability/merged_bodymass_earthsystem.csv")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--out", default="thesis/synthesis/output_volatility_filter")
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--prevalence-threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    decomp = pd.read_csv(args.decomp)
    # Predictor/outcome variables.
    volatility_col = "delta_from_prev_T_field_meanabs"

    # Build per-bin guild occupancy pivot tables.
    diet_occ = pd.read_csv(args.diet_occ)
    mot_occ = pd.read_csv(args.motility_occ)
    hab_occ = pd.read_csv(args.habit_occ)
    diet_p = _pivot_occupancy(diet_occ, cat_col="diet_coarse")
    mot_p = _pivot_occupancy(mot_occ, cat_col="motility_coarse")
    hab_p = _pivot_occupancy(hab_occ, cat_col="habit_coarse")

    merged = decomp.merge(diet_p, on="time_bin", how="left", suffixes=("", ""))
    merged = merged.merge(mot_p, on="time_bin", how="left", suffixes=("", ""))
    merged = merged.merge(hab_p, on="time_bin", how="left", suffixes=("", ""))

    # Define a priori “robust vs specialization” guild sets (marine; coarse PBDB ecospace).
    diet_pos = ["suspension feeder", "detritivore"]
    diet_neg = ["carnivore", "grazer"]
    mot_pos = ["stationary", "slow-moving"]
    mot_neg = ["fast-moving", "facultatively mobile"]
    hab_pos = ["epifaunal"]
    hab_neg = ["planktic"]

    merged["filter_diet"] = _safe_mean(merged, diet_pos) - _safe_mean(merged, diet_neg)
    merged["filter_motility"] = _safe_mean(merged, mot_pos) - _safe_mean(merged, mot_neg)
    merged["filter_habit"] = _safe_mean(merged, hab_pos) - _safe_mean(merged, hab_neg)
    merged["filter_index"] = merged[["filter_diet", "filter_motility", "filter_habit"]].mean(axis=1)

    # Alternative mediator: homogenization via occupancy heterogeneity across categories.
    diet_p2 = diet_p.set_index("time_bin")
    mot_p2 = mot_p.set_index("time_bin")
    hab_p2 = hab_p.set_index("time_bin")
    hd = _heterogeneity_from_pivot(diet_p2, prevalence_threshold=float(args.prevalence_threshold))
    hm = _heterogeneity_from_pivot(mot_p2, prevalence_threshold=float(args.prevalence_threshold))
    hh = _heterogeneity_from_pivot(hab_p2, prevalence_threshold=float(args.prevalence_threshold))
    hetero = pd.DataFrame({"time_bin": hd.index, "hetero_diet": hd.values, "hetero_motility": hm.values, "hetero_habit": hh.values})
    hetero["hetero_mean"] = hetero[["hetero_diet", "hetero_motility", "hetero_habit"]].mean(axis=1)
    hetero["homogenization"] = -hetero["hetero_mean"]
    merged = merged.merge(hetero, on="time_bin", how="left")

    merged = merged.rename(columns={volatility_col: "volatility", "excess_role_js": "convergence"})
    merged.to_csv(out_dir / "merged_marine_filter_convergence.csv", index=False)

    # Core tests (controls: time + sampling proxy).
    results: dict[str, Any] = {
        "inputs": {
            "conv_v2": str(Path(args.conv_v2)),
            "decomp": str(Path(args.decomp)),
            "diet_occ": str(Path(args.diet_occ)),
            "motility_occ": str(Path(args.motility_occ)),
            "habit_occ": str(Path(args.habit_occ)),
            "dino": str(Path(args.dino)),
            "earth": str(Path(args.earth)),
        },
        "controls": ["time_bin", "log1p(n_localities)"],
        "volatility_col": volatility_col,
        "filter_sets": {
            "diet_pos": diet_pos,
            "diet_neg": diet_neg,
            "mot_pos": mot_pos,
            "mot_neg": mot_neg,
            "hab_pos": hab_pos,
            "hab_neg": hab_neg,
        },
    }

    x_vol = merged["volatility"].to_numpy(dtype=float)
    y_conv = merged["convergence"].to_numpy(dtype=float)
    y_filt = merged["filter_index"].to_numpy(dtype=float)
    y_homo = merged["homogenization"].to_numpy(dtype=float)
    y_filt_d = merged["filter_diet"].to_numpy(dtype=float)
    y_filt_m = merged["filter_motility"].to_numpy(dtype=float)
    y_filt_h = merged["filter_habit"].to_numpy(dtype=float)
    t_bin = merged["time_bin"].to_numpy(dtype=float)
    samp = np.log1p(merged["n_localities"].to_numpy(dtype=float))
    controls2 = np.column_stack([t_bin, samp])

    results["H_total_volatility_to_convergence_partial"] = _partial_corr_perm(
        x_vol, y_conv, controls2, permutations=int(args.permutations), seed=int(args.seed) + 1
    )
    results["H_a_volatility_to_filter_partial"] = _partial_corr_perm(
        x_vol, y_filt, controls2, permutations=int(args.permutations), seed=int(args.seed) + 2
    )
    results["H_a_volatility_to_homogenization_partial"] = _partial_corr_perm(
        x_vol, y_homo, controls2, permutations=int(args.permutations), seed=int(args.seed) + 20
    )
    results["H_a_volatility_to_filter_diet_partial"] = _partial_corr_perm(
        x_vol, y_filt_d, controls2, permutations=int(args.permutations), seed=int(args.seed) + 3
    )
    results["H_a_volatility_to_filter_motility_partial"] = _partial_corr_perm(
        x_vol, y_filt_m, controls2, permutations=int(args.permutations), seed=int(args.seed) + 4
    )
    results["H_a_volatility_to_filter_habit_partial"] = _partial_corr_perm(
        x_vol, y_filt_h, controls2, permutations=int(args.permutations), seed=int(args.seed) + 5
    )

    # b path: filter -> convergence controlling volatility + controls.
    controls_b = np.column_stack([controls2, x_vol])
    results["H_b_filter_to_convergence_partial_control_volatility"] = _partial_corr_perm(
        y_filt, y_conv, controls_b, permutations=int(args.permutations), seed=int(args.seed) + 6
    )
    results["H_b_homogenization_to_convergence_partial_control_volatility"] = _partial_corr_perm(
        y_homo, y_conv, controls_b, permutations=int(args.permutations), seed=int(args.seed) + 21
    )

    # Direct effect: volatility -> convergence controlling filter + controls.
    controls_direct = np.column_stack([controls2, y_filt])
    results["H_direct_volatility_to_convergence_partial_control_filter"] = _partial_corr_perm(
        x_vol, y_conv, controls_direct, permutations=int(args.permutations), seed=int(args.seed) + 7
    )
    controls_direct_h = np.column_stack([controls2, y_homo])
    results["H_direct_volatility_to_convergence_partial_control_homogenization"] = _partial_corr_perm(
        x_vol, y_conv, controls_direct_h, permutations=int(args.permutations), seed=int(args.seed) + 22
    )

    # Mediation (bootstrap).
    results["mediation_bootstrap"] = _bootstrap_mediation(
        x_vol,
        y_filt,
        y_conv,
        controls2,
        n_boot=int(args.bootstrap),
        seed=int(args.seed) + 100,
    )
    results["mediation_bootstrap_homogenization"] = _bootstrap_mediation(
        x_vol,
        y_homo,
        y_conv,
        controls2,
        n_boot=int(args.bootstrap),
        seed=int(args.seed) + 110,
    )

    # Mesozoic-only subset (70–200 Ma) to align with dinosaur analysis window.
    meso = merged[(merged["time_bin"] >= 70) & (merged["time_bin"] <= 200)].copy()
    results["mesozoic_n_bins"] = int(len(meso))
    if len(meso) >= 10:
        xv = meso["volatility"].to_numpy(dtype=float)
        yc = meso["convergence"].to_numpy(dtype=float)
        yg = meso["filter_index"].to_numpy(dtype=float)
        tt = meso["time_bin"].to_numpy(dtype=float)
        ss = np.log1p(meso["n_localities"].to_numpy(dtype=float))
        cc = np.column_stack([tt, ss])
        results["meso_H_total_volatility_to_convergence_partial"] = _partial_corr_perm(
            xv, yc, cc, permutations=int(args.permutations), seed=int(args.seed) + 201
        )
        results["meso_H_a_volatility_to_filter_partial"] = _partial_corr_perm(
            xv, yg, cc, permutations=int(args.permutations), seed=int(args.seed) + 202
        )
        yh = meso["homogenization"].to_numpy(dtype=float)
        results["meso_H_a_volatility_to_homogenization_partial"] = _partial_corr_perm(
            xv, yh, cc, permutations=int(args.permutations), seed=int(args.seed) + 204
        )
        results["meso_mediation_bootstrap"] = _bootstrap_mediation(
            xv, yg, yc, cc, n_boot=max(2000, int(args.bootstrap // 4)), seed=int(args.seed) + 203
        )
        results["meso_mediation_bootstrap_homogenization"] = _bootstrap_mediation(
            xv, yh, yc, cc, n_boot=max(2000, int(args.bootstrap // 4)), seed=int(args.seed) + 205
        )

    # Dinosaur “barbell” metric: gap_ratio_hist (Avialae-included, mass2 variant by default).
    dino = pd.read_csv(args.dino)
    dino = dino[(dino["exclude_avialae"] == False) & (dino["mass_variant"] == "mass2")].copy()  # noqa: E712
    dino = dino.dropna(subset=["gap_ratio_hist", volatility_col]).copy()
    dino = dino.rename(columns={volatility_col: "volatility", "time_bin": "time_bin"})

    # Merge dinosaur with marine (overlapping bins) for “co-movement” checks.
    meso_merge = meso.merge(
        dino[["time_bin", "gap_ratio_hist", "bimodality_coeff", "n_specimens", "volatility"]],
        on="time_bin",
        how="inner",
        suffixes=("", "_dino"),
    )
    meso_merge.to_csv(out_dir / "merged_mesozoic_marine_dino.csv", index=False)

    if len(dino) >= 6:
        # Note: within-dinosaur sample is small; treat as qualitative alignment.
        results["dino_corr_volatility_vs_gap_ratio"] = _perm_test_corr(
            dino["volatility"].to_numpy(dtype=float),
            dino["gap_ratio_hist"].to_numpy(dtype=float),
            permutations=int(args.permutations),
            seed=int(args.seed) + 301,
        )

    if len(meso_merge) >= 6:
        results["meso_corr_marine_convergence_vs_dino_gap_ratio"] = _perm_test_corr(
            meso_merge["convergence"].to_numpy(dtype=float),
            meso_merge["gap_ratio_hist"].to_numpy(dtype=float),
            permutations=int(args.permutations),
            seed=int(args.seed) + 401,
        )
        # Shared-driver check: partial out volatility and time.
        controls_cd = np.column_stack(
            [
                meso_merge["time_bin"].to_numpy(dtype=float),
                meso_merge["volatility"].to_numpy(dtype=float),
            ]
        )
        results["meso_partial_corr_convergence_vs_dino_gap_ratio_control_time_volatility"] = _partial_corr_perm(
            meso_merge["convergence"].to_numpy(dtype=float),
            meso_merge["gap_ratio_hist"].to_numpy(dtype=float),
            controls_cd,
            permutations=int(args.permutations),
            seed=int(args.seed) + 402,
        )

    # Robustness: repeat core “volatility → convergence” tests using the original convergence metric (output_v2).
    conv_v2 = pd.read_csv(args.conv_v2)
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})
    v2 = conv_v2.merge(earth, on="time_bin", how="left")
    v2 = v2.merge(merged[["time_bin", "homogenization"]], on="time_bin", how="left")
    v2 = v2.dropna(subset=["functional_excess_similarity_js", volatility_col, "homogenization"]).copy()
    v2.to_csv(out_dir / "merged_convergence_v2_homogenization.csv", index=False)

    xv2 = v2[volatility_col].to_numpy(dtype=float)
    yv2 = v2["functional_excess_similarity_js"].to_numpy(dtype=float)
    hv2 = v2["homogenization"].to_numpy(dtype=float)
    tv2 = v2["time_bin"].to_numpy(dtype=float)
    sv2 = np.log1p(v2["n_localities"].to_numpy(dtype=float))
    ctrl_v2 = np.column_stack([tv2, sv2])
    results["v2_total_volatility_to_convergence_partial"] = _partial_corr_perm(
        xv2, yv2, ctrl_v2, permutations=int(args.permutations), seed=int(args.seed) + 501
    )
    results["v2_a_volatility_to_homogenization_partial"] = _partial_corr_perm(
        xv2, hv2, ctrl_v2, permutations=int(args.permutations), seed=int(args.seed) + 502
    )
    ctrl_v2_b = np.column_stack([ctrl_v2, xv2])
    results["v2_b_homogenization_to_convergence_partial_control_volatility"] = _partial_corr_perm(
        hv2, yv2, ctrl_v2_b, permutations=int(args.permutations), seed=int(args.seed) + 503
    )
    ctrl_v2_direct = np.column_stack([ctrl_v2, hv2])
    results["v2_direct_volatility_to_convergence_partial_control_homogenization"] = _partial_corr_perm(
        xv2, yv2, ctrl_v2_direct, permutations=int(args.permutations), seed=int(args.seed) + 504
    )
    results["v2_mediation_bootstrap_homogenization"] = _bootstrap_mediation(
        xv2, hv2, yv2, ctrl_v2, n_boot=max(2000, int(args.bootstrap // 4)), seed=int(args.seed) + 505
    )

    v2_meso = v2[(v2["time_bin"] >= 70) & (v2["time_bin"] <= 200)].copy()
    results["v2_mesozoic_n_bins"] = int(len(v2_meso))
    if len(v2_meso) >= 10:
        xv = v2_meso[volatility_col].to_numpy(dtype=float)
        yc = v2_meso["functional_excess_similarity_js"].to_numpy(dtype=float)
        hg = v2_meso["homogenization"].to_numpy(dtype=float)
        tt = v2_meso["time_bin"].to_numpy(dtype=float)
        ss = np.log1p(v2_meso["n_localities"].to_numpy(dtype=float))
        cc = np.column_stack([tt, ss])
        results["v2_meso_total_volatility_to_convergence_partial"] = _partial_corr_perm(
            xv, yc, cc, permutations=int(args.permutations), seed=int(args.seed) + 520
        )
        results["v2_meso_a_volatility_to_homogenization_partial"] = _partial_corr_perm(
            xv, hg, cc, permutations=int(args.permutations), seed=int(args.seed) + 521
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Figures (marine, full interval).
    _plot_scatter(
        merged,
        x="volatility",
        y="convergence",
        out_path=fig_dir / "scatter_volatility_convergence.png",
        title="Marine ecospace: volatility vs convergence (excess role JS)",
    )
    _plot_scatter(
        merged,
        x="volatility",
        y="filter_index",
        out_path=fig_dir / "scatter_volatility_filter.png",
        title="Marine ecospace: volatility vs filter index",
    )
    _plot_scatter(
        merged,
        x="filter_index",
        y="convergence",
        out_path=fig_dir / "scatter_filter_convergence.png",
        title="Marine ecospace: filter index vs convergence",
    )
    _plot_scatter(
        merged,
        x="volatility",
        y="homogenization",
        out_path=fig_dir / "scatter_volatility_homogenization.png",
        title="Marine ecospace: volatility vs homogenization (−heterogeneity)",
    )
    _plot_scatter(
        merged,
        x="homogenization",
        y="convergence",
        out_path=fig_dir / "scatter_homogenization_convergence.png",
        title="Marine ecospace: homogenization vs convergence",
    )
    _plot_timeseries(merged, out_path=fig_dir / "timeseries_marine.png", title="Marine ecospace: volatility, filter, convergence")

    if len(meso) >= 6:
        _plot_timeseries(meso, out_path=fig_dir / "timeseries_mesozoic.png", title="Mesozoic slice (70–200 Ma): marine metrics")

    if len(meso_merge) >= 6:
        _plot_scatter(
            meso_merge,
            x="volatility",
            y="gap_ratio_hist",
            out_path=fig_dir / "scatter_volatility_dino_gap.png",
            title="Dinosaurs: volatility vs gap_ratio_hist (Avialae-included, mass2)",
        )
        _plot_scatter(
            meso_merge,
            x="convergence",
            y="gap_ratio_hist",
            out_path=fig_dir / "scatter_marine_convergence_vs_dino_gap.png",
            title="Mesozoic: marine convergence vs dinosaur gap_ratio_hist",
        )

    # Human-readable summary.
    def _fmt(d: dict[str, Any] | None) -> str:
        if not d:
            return "corr=nan, p=nan, n=0"
        return "corr={c:.3f}, perm-p={p:.3g}, n={n}".format(
            c=float(d.get("corr", float("nan"))),
            p=float(d.get("p_perm", float("nan"))),
            n=int(d.get("n") or 0),
        )

    med = results.get("mediation_bootstrap", {})
    med_h = results.get("mediation_bootstrap_homogenization", {})
    lines = [
        "# Synthesis test: volatility-as-filter (marine convergence) + alignment with dinosaur barbell metric",
        "",
        "Marine ecospace test (PBDB ecospace roles, locality grid + 10 Myr bins):",
        "- volatility proxy: CESM |ΔT| field mean absolute change (Li et al. 2022)",
        "- convergence: excess similarity of full role composition (JS residual vs taxonomic similarity)",
        "- mediator: filter index from a priori guild occupancy sets (robust minus specialized strategies)",
        "",
        "## Core results (controls: time + log1p(n_localities))",
        "",
        f"- Total: volatility → convergence: {_fmt(results.get('H_total_volatility_to_convergence_partial'))}",
        f"- Path a: volatility → filter index: {_fmt(results.get('H_a_volatility_to_filter_partial'))}",
        f"- Path b: filter → convergence (controls volatility): {_fmt(results.get('H_b_filter_to_convergence_partial_control_volatility'))}",
        f"- Direct: volatility → convergence (controls filter): {_fmt(results.get('H_direct_volatility_to_convergence_partial_control_filter'))}",
        "",
        "## Mediation (standardized OLS; bootstrap 95% CI)",
        "",
        f"- n={med.get('n')}, a={med.get('a'):.3f}, b={med.get('b'):.3f}, direct={med.get('direct'):.3f}, indirect={med.get('indirect'):.3f}",
    ]
    ci = (med.get("ci") or {}).get("indirect")
    if ci:
        lines.append(f"- indirect 95% CI: [{float(ci[0]):.3f}, {float(ci[1]):.3f}]")

    lines.extend(
        [
            "",
            "## Alternative mediator: homogenization (−mean p(1−p) across prevalent categories)",
            "",
            f"- Path a: volatility → homogenization: {_fmt(results.get('H_a_volatility_to_homogenization_partial'))}",
            f"- Path b: homogenization → convergence (controls volatility): {_fmt(results.get('H_b_homogenization_to_convergence_partial_control_volatility'))}",
            f"- Direct: volatility → convergence (controls homogenization): {_fmt(results.get('H_direct_volatility_to_convergence_partial_control_homogenization'))}",
            "",
            f"- mediation (homogenization): n={med_h.get('n')}, a={med_h.get('a'):.3f}, b={med_h.get('b'):.3f}, "
            f"direct={med_h.get('direct'):.3f}, indirect={med_h.get('indirect'):.3f}",
        ]
    )
    ci_h = (med_h.get("ci") or {}).get("indirect")
    if ci_h:
        lines.append(f"- indirect 95% CI: [{float(ci_h[0]):.3f}, {float(ci_h[1]):.3f}]")

    lines.extend(
        [
            "",
            "## Robustness: original convergence metric (PBDB ecospace v2 output)",
            "",
            f"- Total: volatility → convergence_v2: {_fmt(results.get('v2_total_volatility_to_convergence_partial'))}",
            f"- Path a: volatility → homogenization: {_fmt(results.get('v2_a_volatility_to_homogenization_partial'))}",
            f"- Path b: homogenization → convergence_v2 (controls volatility): {_fmt(results.get('v2_b_homogenization_to_convergence_partial_control_volatility'))}",
            f"- Direct: volatility → convergence_v2 (controls homogenization): {_fmt(results.get('v2_direct_volatility_to_convergence_partial_control_homogenization'))}",
        ]
    )
    med_v2 = results.get("v2_mediation_bootstrap_homogenization", {})
    if med_v2 and med_v2.get("n"):
        ci_v2 = (med_v2.get("ci") or {}).get("indirect")
        lines.append(
            f"- mediation via homogenization: n={med_v2.get('n')}, indirect={float(med_v2.get('indirect', float('nan'))):.3f}, "
            f"95% CI={ci_v2}"
        )
    if "v2_meso_total_volatility_to_convergence_partial" in results:
        lines.extend(
            [
                "",
                "### Mesozoic slice (70–200 Ma; v2 metric)",
                "",
                f"- volatility → convergence_v2: {_fmt(results.get('v2_meso_total_volatility_to_convergence_partial'))}",
                f"- volatility → homogenization: {_fmt(results.get('v2_meso_a_volatility_to_homogenization_partial'))}",
            ]
        )

    if "meso_H_total_volatility_to_convergence_partial" in results:
        lines.extend(
            [
                "",
                "## Mesozoic slice (70–200 Ma)",
                "",
                f"- volatility → convergence: {_fmt(results.get('meso_H_total_volatility_to_convergence_partial'))}",
                f"- volatility → filter index: {_fmt(results.get('meso_H_a_volatility_to_filter_partial'))}",
                f"- volatility → homogenization: {_fmt(results.get('meso_H_a_volatility_to_homogenization_partial'))}",
            ]
        )
        med2 = results.get("meso_mediation_bootstrap", {})
        if med2 and med2.get("n"):
            lines.append(
                f"- mediation (n={med2.get('n')}): indirect={float(med2.get('indirect', float('nan'))):.3f}, "
                f"95% CI={med2.get('ci', {}).get('indirect')}"
            )
        med3 = results.get("meso_mediation_bootstrap_homogenization", {})
        if med3 and med3.get("n"):
            lines.append(
                f"- mediation via homogenization (n={med3.get('n')}): indirect={float(med3.get('indirect', float('nan'))):.3f}, "
                f"95% CI={med3.get('ci', {}).get('indirect')}"
            )

    if "dino_corr_volatility_vs_gap_ratio" in results:
        lines.extend(
            [
                "",
                "## Dinosaur alignment check (Avialae-included, mass2; small n)",
                "",
                f"- volatility → gap_ratio_hist: {_fmt(results.get('dino_corr_volatility_vs_gap_ratio'))}",
            ]
        )
    if "meso_corr_marine_convergence_vs_dino_gap_ratio" in results:
        lines.append(f"- marine convergence ↔ dinosaur gap_ratio_hist: {_fmt(results.get('meso_corr_marine_convergence_vs_dino_gap_ratio'))}")
        lines.append(
            f"- partial corr (controls time+volatility): {_fmt(results.get('meso_partial_corr_convergence_vs_dino_gap_ratio_control_time_volatility'))}"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Marine merged table: `{out_dir / 'merged_marine_filter_convergence.csv'}`",
            f"- Mesozoic marine+dino: `{out_dir / 'merged_mesozoic_marine_dino.csv'}`",
            f"- Stats: `{out_dir / 'analysis_results.json'}`",
            f"- Figures: `{fig_dir}`",
            "",
            "## Interpretation guardrails",
            "",
            "- Time bins are autocorrelated; permutation/bootstrap here treat bins as exchangeable (use as hypothesis test, not final inference).",
            "- The filter index uses coarse ecospace categories and should be stress-tested with alternative sets and sampling controls (PBDB collections, Macrostrat rock area).",
            "",
        ]
    )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
