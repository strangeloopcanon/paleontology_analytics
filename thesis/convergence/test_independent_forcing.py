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


def _perm_test_corr(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 6:
        return {"corr": float("nan"), "p_perm": float("nan"), "n": float(len(x))}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(int(permutations)):
        yp = rng.permutation(y)
        c = float(np.corrcoef(x, yp)[0, 1])
        if abs(c) >= abs(obs):
            more_extreme += 1
    p = (more_extreme + 1) / (int(permutations) + 1)
    return {"corr": obs, "p_perm": float(p), "n": float(len(x))}


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    x = x.astype(float)
    mask = np.isfinite(y) & np.isfinite(x)
    yy = y[mask]
    xx = x[mask]
    if len(yy) < 3:
        out = np.full_like(y, fill_value=np.nan, dtype=float)
        return out
    A = np.column_stack([np.ones(len(xx)), xx])
    coef, _, _, _ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A.dot(coef)
    out = np.full_like(y, fill_value=np.nan, dtype=float)
    out[mask] = resid
    return out


def _partial_corr_perm(x: np.ndarray, y: np.ndarray, control: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    rx = _residualize(x, control)
    ry = _residualize(y, control)
    return _perm_test_corr(rx, ry, permutations=int(permutations), seed=int(seed))


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y]].dropna()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.scatter(d[x], d[y], alpha=0.75, s=35, color="#1f77b4", edgecolors="none")
    A = np.vstack([d[x].to_numpy(), np.ones(len(d))]).T
    coef, _, _, _ = np.linalg.lstsq(A, d[y].to_numpy(), rcond=None)
    xx = np.linspace(float(d[x].min()), float(d[x].max()), 60)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="black", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--convergence", default="thesis/convergence/output_v2/timebin_metrics.csv")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--out", default="thesis/convergence/output_independent_forcing")
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    conv = pd.read_csv(args.convergence)
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})

    merged = conv.merge(earth, on="time_bin", how="left")
    merged = merged.dropna(subset=["functional_excess_similarity_js"]).copy()
    merged.to_csv(out_dir / "merged_convergence_earthsystem.csv", index=False)

    y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
    t = merged["time_bin"].to_numpy(dtype=float)

    predictors = [
        "delta_from_prev_T_global_abs",
        "delta_from_prev_T_field_meanabs",
        "delta_from_prev_landfrac_field_meanabs",
        "delta_from_prev_coastline_abs",
        "delta_from_prev_land_components_abs",
    ]

    results: dict[str, Any] = {"n_bins": int(len(merged))}
    for j, col in enumerate(predictors):
        x = merged[col].to_numpy(dtype=float)
        results[f"corr_{col}"] = _perm_test_corr(x, y, permutations=int(args.permutations), seed=int(args.seed) + j)
        results[f"partial_corr_{col}_control_time"] = _partial_corr_perm(
            x, y, t, permutations=int(args.permutations), seed=int(args.seed) + 100 + j
        )
        _plot_scatter(
            merged,
            x=col,
            y="functional_excess_similarity_js",
            out_path=fig_dir / f"scatter_{col}.png",
            title=f"Convergence (JS residual) vs {col}",
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Summary markdown.
    lines = [
        "# Independent forcing test: convergence vs CESM-derived volatility",
        "",
        "This merges PBDB ecospace convergence metrics with an independent CESM snapshot series (Li et al. 2022) and tests whether",
        "functional convergence tracks climate and/or paleogeography volatility.",
        "",
        f"- Convergence bins: `{Path(args.convergence)}`",
        f"- Earth-system bins: `{Path(args.earth)}`",
        f"- Merged: `{out_dir / 'merged_convergence_earthsystem.csv'}`",
        f"- Permutations: {int(args.permutations)}",
        "",
        "## Results (correlation; permutation p-values)",
        "",
        "| Predictor | corr | perm-p | partial corr (| time) | perm-p | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for col in predictors:
        r = results.get(f"corr_{col}", {})
        pr = results.get(f"partial_corr_{col}_control_time", {})
        lines.append(
            "| {col} | {c:.3f} | {p:.3g} | {pc:.3f} | {pp:.3g} | {n:d} |".format(
                col=col,
                c=float(r.get("corr")) if np.isfinite(r.get("corr", float("nan"))) else float("nan"),
                p=float(r.get("p_perm")) if np.isfinite(r.get("p_perm", float("nan"))) else float("nan"),
                pc=float(pr.get("corr")) if np.isfinite(pr.get("corr", float("nan"))) else float("nan"),
                pp=float(pr.get("p_perm")) if np.isfinite(pr.get("p_perm", float("nan"))) else float("nan"),
                n=int(r.get("n") or 0),
            )
        )
    lines.extend(["", "## Figures", "", f"- `{fig_dir}`", ""])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()

