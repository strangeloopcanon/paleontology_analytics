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


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y, "n_specimens"]].dropna()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(d[x], d[y], s=np.clip(d["n_specimens"] * 1.5, 25, 200), alpha=0.75, color="#2ca02c", edgecolors="none")
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
    p.add_argument("--body", default="thesis/body_size_stability/output/body_mass_timebins.csv")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--out", default="thesis/body_size_stability/output_independent_stability")
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    body = pd.read_csv(args.body)
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})

    merged = body.merge(earth, on="time_bin", how="left")
    merged.to_csv(out_dir / "merged_bodymass_earthsystem.csv", index=False)

    predictors = [
        "delta_from_prev_T_global_abs",
        "delta_from_prev_T_field_meanabs",
        "delta_from_prev_T_coherence_ratio",
        "delta_from_prev_T_sign_agreement_frac",
        "delta_from_prev_T_effective_rank",
        "delta_from_prev_landfrac_field_meanabs",
        "delta_from_prev_coastline_abs",
    ]
    outcomes = ["bimodality_coeff", "gap_ratio_hist"]

    results: list[dict[str, Any]] = []
    for (exclude_avialae, mass_variant), sub in merged.groupby(["exclude_avialae", "mass_variant"], sort=False):
        row: dict[str, Any] = {"exclude_avialae": bool(exclude_avialae), "mass_variant": str(mass_variant)}
        for ycol in outcomes:
            for xcol in predictors:
                r = _perm_test_corr(
                    sub[xcol].to_numpy(dtype=float),
                    sub[ycol].to_numpy(dtype=float),
                    permutations=int(args.permutations),
                    seed=int(args.seed) + hash((exclude_avialae, mass_variant, ycol, xcol)) % 10000,
                )
                row[f"corr_{xcol}_vs_{ycol}"] = r
        results.append(row)

        # Plots (bimodality only, most stable).
        for xcol in predictors:
            _plot_scatter(
                sub,
                x=xcol,
                y="bimodality_coeff",
                out_path=fig_dir / f"scatter_{xcol}_bimodality_exclAvialae_{int(exclude_avialae)}_{mass_variant}.png",
                title=f"{xcol} vs bimodality (excl Avialae={exclude_avialae}, {mass_variant})",
            )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Summary markdown.
    lines = [
        "# Independent forcing test: dinosaur body-size structure vs Earth-system volatility",
        "",
        "This merges the Benson et al. (2014) dinosaur body-mass time bins with an independent CESM snapshot series (Li et al. 2022) and tests",
        "whether the “missing middle sizes” metrics covary with climate/paleogeography volatility.",
        "",
        f"- Body bins: `{Path(args.body)}`",
        f"- Earth-system bins: `{Path(args.earth)}`",
        f"- Merged: `{out_dir / 'merged_bodymass_earthsystem.csv'}`",
        f"- Permutations: {int(args.permutations)}",
        "",
        "## Notes",
        "",
        "- Signs: higher `delta_from_prev_*` means *more volatility* between adjacent 10 Myr CESM snapshots.",
        "- If “stability fosters missing-middle bimodality”, we would expect **negative** correlations between volatility and bimodality.",
        "",
        "## Files",
        "",
        f"- Results JSON: `{out_dir / 'analysis_results.json'}`",
        f"- Figures: `{fig_dir}`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
