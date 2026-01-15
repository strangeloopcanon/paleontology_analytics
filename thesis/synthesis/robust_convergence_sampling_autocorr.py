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


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, fill_value=np.nan, dtype=float)
    if int(np.sum(mask)) < (X.shape[1] + 3):
        return out
    yy = y[mask]
    XX = X[mask]
    A = np.column_stack([np.ones(len(XX)), XX])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    out[mask] = yy - A.dot(beta)
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _pca_scores(X: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD on standardized columns.

    Returns:
      scores: (n, k) component scores
      explained: (k,) explained variance fractions
      loadings: (k, p) component loadings in standardized feature space
    """
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


def _iid_perm_p(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 6:
        return {"corr": float("nan"), "p_perm": float("nan"), "n": float(n)}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(int(seed))
    more = 0
    for _ in range(int(permutations)):
        yp = rng.permutation(y)
        c = float(np.corrcoef(x, yp)[0, 1])
        if abs(c) >= abs(obs):
            more += 1
    p = (more + 1) / (int(permutations) + 1)
    return {"corr": float(obs), "p_perm": float(p), "n": float(n)}


def _circular_shift_p(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    # Circularly shift y relative to x to preserve autocorrelation structure.
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 6:
        return {"corr": float("nan"), "p_shift": float("nan"), "n": float(n)}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(int(seed))
    shifts = rng.integers(1, n, size=int(permutations))
    more = 0
    for s in shifts:
        ys = np.roll(y, int(s))
        c = float(np.corrcoef(x, ys)[0, 1])
        if abs(c) >= abs(obs):
            more += 1
    p = (more + 1) / (int(permutations) + 1)
    return {"corr": float(obs), "p_shift": float(p), "n": float(n)}


def _classify_env(env: Any) -> str:
    if env is None or (isinstance(env, float) and not np.isfinite(env)):
        return "unknown"
    s = str(env).strip().lower()
    if not s:
        return "unknown"
    terrestrial_terms = [
        "terrestrial",
        "fluvial",
        "lacustrine",
        "delta",
        "freshwater",
        "non-marine",
        "nonmarine",
        "eolian",
        "loess",
        "soil",
        "cave",
        "spring",
        "swamp",
        "paludal",
        "floodplain",
        "karst",
    ]
    if any(t in s for t in terrestrial_terms):
        return "terrestrial"
    marine_terms = [
        "marine",
        "reef",
        "subtidal",
        "offshore",
        "shelf",
        "basinal",
        "slope",
        "lagoon",
        "open",
        "deep",
        "carbonate",
        "platform",
        "pelagic",
        "ocean",
        "intertidal",
        "coastal",
    ]
    if any(t in s for t in marine_terms):
        return "marine"
    return "other"


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y]].dropna()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    ax.scatter(d[x], d[y], alpha=0.75, s=40, color="#1f77b4", edgecolors="none")
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
    p.add_argument("--convergence", default="thesis/convergence/output_v2/timebin_metrics.csv")
    p.add_argument("--pbdb-extended", default="data/processed/pbdb_occurrences_extended.parquet")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument(
        "--macrostrat",
        default="data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv",
        help="Optional Macrostrat rock-record proxy time series (binned).",
    )
    p.add_argument("--out", default="thesis/synthesis/output_convergence_sampling_autocorr")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=77)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    conv = pd.read_csv(args.convergence)
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})
    macro = None
    macro_path = Path(args.macrostrat)
    if macro_path.exists():
        macro = pd.read_csv(macro_path)
    pb = pd.read_parquet(
        args.pbdb_extended,
        columns=[
            "occurrence_no",
            "collection_no",
            "reference_no",
            "mid_ma",
            "environment",
        ],
    )

    # Compute sampling proxies per time bin.
    pb = pb.dropna(subset=["mid_ma"]).copy()
    pb["time_bin"] = (pd.to_numeric(pb["mid_ma"], errors="coerce") / float(args.time_bin_myr)).round() * float(
        args.time_bin_myr
    )
    pb["env_class"] = pb["environment"].map(_classify_env)

    def _agg(sub: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (
            sub.groupby("time_bin")
            .agg(
                **{
                    f"{prefix}n_occurrences": ("occurrence_no", lambda s: int(pd.Series(s).dropna().nunique())),
                    f"{prefix}n_collections": ("collection_no", lambda s: int(pd.Series(s).dropna().nunique())),
                    f"{prefix}n_references": ("reference_no", lambda s: int(pd.Series(s).dropna().nunique())),
                }
            )
            .reset_index()
        )

    total = _agg(pb, prefix="")
    marine = _agg(pb[pb["env_class"] == "marine"], prefix="marine_")
    terr = _agg(pb[pb["env_class"] == "terrestrial"], prefix="terr_")

    samp = total.merge(marine, on="time_bin", how="left").merge(terr, on="time_bin", how="left")

    merged = conv.merge(earth, on="time_bin", how="left").merge(samp, on="time_bin", how="left")
    if macro is not None and "time_bin" in macro.columns:
        merged = merged.merge(macro, on="time_bin", how="left")
    merged = merged.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    merged.to_csv(out_dir / "merged.csv", index=False)

    # Primary variables.
    y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
    v = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    t = merged["time_bin"].to_numpy(dtype=float)
    nloc = np.log1p(merged["n_localities"].to_numpy(dtype=float))
    ncoll = np.log1p(merged["marine_n_collections"].to_numpy(dtype=float))
    nocc = np.log1p(merged["marine_n_occurrences"].to_numpy(dtype=float))
    prov = merged["provinciality"].to_numpy(dtype=float)

    # Optional Macrostrat covariates (rock record proxy).
    # Note: these are typically highly collinear with PBDB sampling proxies; treat as sensitivity checks.
    macro_area = None
    macro_sections = None
    if macro is not None and "macro_col_area_sum" in merged.columns:
        macro_area = np.log1p(merged["macro_col_area_sum"].to_numpy(dtype=float))
    if macro is not None and "macro_n_sections" in merged.columns:
        macro_sections = np.log1p(merged["macro_n_sections"].to_numpy(dtype=float))

    # PCA sampling index (handles collinearity between PBDB sampling proxies and Macrostrat proxies).
    sampling_feature_names = ["log1p(n_localities)", "log1p(marine_n_collections)", "log1p(marine_n_occurrences)"]
    sampling_features = [nloc, ncoll, nocc]
    if macro_area is not None:
        sampling_feature_names.append("log1p(macro_col_area_sum)")
        sampling_features.append(macro_area)
    if macro_sections is not None:
        sampling_feature_names.append("log1p(macro_n_sections)")
        sampling_features.append(macro_sections)

    sampling_matrix = np.column_stack(sampling_features)
    pcs, pc_expl, pc_load = _pca_scores(sampling_matrix, k=2)
    merged["sampling_pc1"] = pcs[:, 0]
    merged["sampling_pc2"] = pcs[:, 1]
    (out_dir / "sampling_pca.json").write_text(
        json.dumps(
            {
                "feature_names": sampling_feature_names,
                "explained_variance": [float(x) for x in pc_expl],
                "loadings": [[float(v) for v in row] for row in pc_load],
            },
            indent=2,
        )
        + "\n"
    )

    # Controls sets.
    controls_basic = np.column_stack([t])
    controls_loc = np.column_stack([t, nloc])
    controls_sampling = np.column_stack([t, nloc, ncoll, nocc])
    controls_full = np.column_stack([t, nloc, ncoll, nocc, prov])
    controls_full_macro_area = (
        np.column_stack([t, nloc, ncoll, nocc, prov, macro_area]) if macro_area is not None else None
    )
    controls_full_macro_sections = (
        np.column_stack([t, nloc, ncoll, nocc, prov, macro_sections]) if macro_sections is not None else None
    )
    controls_full_macro_both = (
        np.column_stack([t, nloc, ncoll, nocc, prov, macro_area, macro_sections])
        if (macro_area is not None and macro_sections is not None)
        else None
    )
    controls_pc1 = np.column_stack([t, pcs[:, 0]])
    controls_pc1_prov = np.column_stack([t, pcs[:, 0], prov])
    controls_pc12_prov = np.column_stack([t, pcs[:, 0], pcs[:, 1], prov])

    def _partial(x: np.ndarray, y_: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rx = _residualize(x, controls)
        ry = _residualize(y_, controls)
        # keep original order; circular shift assumes order is meaningful (time-sorted already in file).
        return rx, ry

    results: dict[str, Any] = {"n_bins": int(len(merged))}
    configs = [
        ("control_time", controls_basic),
        ("control_time_loc", controls_loc),
        ("control_time_loc_coll_occ", controls_sampling),
        ("control_time_loc_coll_occ_prov", controls_full),
        ("control_time_sampling_pc1", controls_pc1),
        ("control_time_sampling_pc1_prov", controls_pc1_prov),
        ("control_time_sampling_pc12_prov", controls_pc12_prov),
    ]
    if controls_full_macro_area is not None:
        configs.append(("control_time_loc_coll_occ_prov_macro_area", controls_full_macro_area))
    if controls_full_macro_sections is not None:
        configs.append(("control_time_loc_coll_occ_prov_macro_sections", controls_full_macro_sections))
    if controls_full_macro_both is not None:
        configs.append(("control_time_loc_coll_occ_prov_macro_area_sections", controls_full_macro_both))
    for i, (name, ctrl) in enumerate(configs):
        rx, ry = _partial(v, y, ctrl)
        results[f"{name}_iid_perm"] = _iid_perm_p(rx, ry, permutations=int(args.permutations), seed=int(args.seed) + i)
        results[f"{name}_circular_shift"] = _circular_shift_p(
            rx, ry, permutations=int(args.permutations), seed=int(args.seed) + 100 + i
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Quick plots.
    _plot_scatter(
        merged,
        x="delta_from_prev_T_field_meanabs",
        y="functional_excess_similarity_js",
        out_path=fig_dir / "scatter_volatility_vs_convergence.png",
        title="Volatility vs convergence (raw bins)",
    )
    _plot_scatter(
        merged,
        x="marine_n_collections",
        y="functional_excess_similarity_js",
        out_path=fig_dir / "scatter_marine_collections_vs_convergence.png",
        title="Marine sampling proxy vs convergence",
    )

    # Summary markdown.
    def _fmt(entry: dict[str, Any] | None, p_key: str) -> str:
        if not entry:
            return "corr=nan, p=nan, n=0"
        return "corr={c:.3f}, p={p:.3g}, n={n}".format(
            c=float(entry.get("corr", float("nan"))),
            p=float(entry.get(p_key, float("nan"))),
            n=int(entry.get("n") or 0),
        )

    lines = [
        "# Robustness: convergence vs volatility with sampling + autocorrelation-aware tests",
        "",
        "We merge:",
        f"- Convergence bins: `{Path(args.convergence)}`",
        f"- Extended PBDB occurrences (for sampling proxies): `{Path(args.pbdb_extended)}`",
        f"- Independent forcing: `{Path(args.earth)}`",
        f"- Macrostrat proxies: `{macro_path}`" if macro is not None else "- Macrostrat proxies: (not found; skipped)",
        f"- Sampling PCA features: {', '.join(sampling_feature_names)}",
        f"- Sampling PCA PC1 explained variance: {float(pc_expl[0]):.3f}",
        "",
        "Volatility predictor: `delta_from_prev_T_field_meanabs` (CESM).",
        "Convergence outcome: `functional_excess_similarity_js` (PBDB ecospace v2).",
        "",
        "## Partial correlation tests",
        "",
        "- IID permutation p-values shuffle residuals (exchangeable bins).",
        "- Circular-shift p-values preserve autocorrelation structure (time-ordered bins).",
        "",
    ]
    for name, _ in configs:
        iid = results.get(f"{name}_iid_perm", {})
        shift = results.get(f"{name}_circular_shift", {})
        lines.append(f"- {name}: iid({_fmt(iid,'p_perm')}); shift({_fmt(shift,'p_shift')})")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Merged table: `{out_dir / 'merged.csv'}`",
            f"- Stats: `{out_dir / 'analysis_results.json'}`",
            f"- Figures: `{fig_dir}`",
            "",
            "## Notes",
            "",
            "- Sampling proxies are derived from PBDB `collection_no` and a coarse environment classifier on PBDB `environment` strings; treat as approximate.",
            "- For final inference, prefer explicit time-series models or block bootstraps, and integrate Macrostrat/rock-area covariates.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
