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


def _clean_name(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


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
    p.add_argument("--convergence", required=True, help="timebin_metrics.csv from run_convergence_analysis_occ_ecospace.py")
    p.add_argument("--pbdb-csv", required=True, help="Raw PBDB occs/list.csv export used for sampling proxies.")
    p.add_argument("--env-substr", default="terrestrial", help="Substring filter on `taxon_environment` for sampling proxies.")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--out", required=True)
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

    pb = pd.read_csv(
        args.pbdb_csv,
        usecols=["occurrence_no", "collection_no", "reference_no", "max_ma", "min_ma", "genus", "taxon_environment"],
        low_memory=False,
    )
    pb["genus"] = pb["genus"].map(_clean_name)
    pb["taxon_environment"] = pb["taxon_environment"].map(_clean_name)
    pb["max_ma"] = pd.to_numeric(pb["max_ma"], errors="coerce")
    pb["min_ma"] = pd.to_numeric(pb["min_ma"], errors="coerce")
    pb = pb.dropna(subset=["occurrence_no", "collection_no", "reference_no", "max_ma", "min_ma"]).copy()
    pb["mid_ma"] = (pb["max_ma"] + pb["min_ma"]) / 2.0
    pb["time_bin"] = (pd.to_numeric(pb["mid_ma"], errors="coerce") / float(args.time_bin_myr)).round() * float(
        args.time_bin_myr
    )
    env_sub = str(args.env_substr).strip().lower()
    if env_sub:
        pb = pb[pb["taxon_environment"].astype(str).str.lower().str.contains(env_sub, na=False)].copy()

    samp = (
        pb.groupby("time_bin")
        .agg(
            n_occurrences=("occurrence_no", lambda s: int(pd.Series(s).dropna().nunique())),
            n_collections=("collection_no", lambda s: int(pd.Series(s).dropna().nunique())),
            n_references=("reference_no", lambda s: int(pd.Series(s).dropna().nunique())),
            n_genera=("genus", lambda s: int(pd.Series(s).dropna().nunique())),
        )
        .reset_index()
    )

    merged = conv.merge(earth, on="time_bin", how="left").merge(samp, on="time_bin", how="left")
    merged = merged.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    merged.to_csv(out_dir / "merged.csv", index=False)

    y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
    v = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    t = merged["time_bin"].to_numpy(dtype=float)
    nloc = np.log1p(merged["n_localities"].to_numpy(dtype=float))
    ncoll = np.log1p(merged["n_collections"].to_numpy(dtype=float))
    nocc = np.log1p(merged["n_occurrences"].to_numpy(dtype=float))
    prov = merged["provinciality"].to_numpy(dtype=float)

    controls_basic = np.column_stack([t])
    controls_loc = np.column_stack([t, nloc])
    controls_sampling = np.column_stack([t, nloc, ncoll, nocc])
    controls_full = np.column_stack([t, nloc, ncoll, nocc, prov])

    configs = [
        ("control_time", controls_basic),
        ("control_time_loc", controls_loc),
        ("control_time_loc_coll_occ", controls_sampling),
        ("control_time_loc_coll_occ_prov", controls_full),
    ]

    results: dict[str, Any] = {"n_bins": int(len(merged))}
    for i, (name, ctrl) in enumerate(configs):
        rx = _residualize(v, ctrl)
        ry = _residualize(y, ctrl)
        results[f"{name}_iid_perm"] = _iid_perm_p(rx, ry, permutations=int(args.permutations), seed=int(args.seed) + i)
        results[f"{name}_circular_shift"] = _circular_shift_p(
            rx, ry, permutations=int(args.permutations), seed=int(args.seed) + 100 + i
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    _plot_scatter(
        merged,
        x="delta_from_prev_T_field_meanabs",
        y="functional_excess_similarity_js",
        out_path=fig_dir / "scatter_volatility_vs_convergence.png",
        title="Volatility vs convergence (raw bins)",
    )

    def _fmt(entry: dict[str, Any] | None, p_key: str) -> str:
        if not entry:
            return "corr=nan, p=nan, n=0"
        return "corr={c:.3f}, p={p:.3g}, n={n}".format(
            c=float(entry.get("corr", float("nan"))),
            p=float(entry.get(p_key, float("nan"))),
            n=int(entry.get("n") or 0),
        )

    lines = [
        "# Robustness (occ-ecospace): convergence vs volatility with sampling + autocorrelation-aware tests",
        "",
        "We merge:",
        f"- Convergence bins: `{Path(args.convergence)}`",
        f"- PBDB occs export (sampling proxies): `{Path(args.pbdb_csv)}`",
        f"- Independent forcing: `{Path(args.earth)}`",
        "",
        f"Taxon-environment filter for sampling proxies: `{args.env_substr}` (substring on `taxon_environment`).",
        "",
        "Volatility predictor: `delta_from_prev_T_field_meanabs` (CESM).",
        "Convergence outcome: `functional_excess_similarity_js` (occ-level ecospace JS residual).",
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
            "- Sampling proxies (`n_occurrences`, `n_collections`) are computed from the same PBDB occs export used to compute convergence.",
            "- For publication-grade inference, prefer pair-level or hierarchical models and explicit time-series error structures.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()

