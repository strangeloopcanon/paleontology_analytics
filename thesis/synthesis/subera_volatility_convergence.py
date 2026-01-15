from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
        c = float(np.corrcoef(x, np.roll(y, int(s)))[0, 1])
        if abs(c) >= abs(obs):
            more += 1
    p = (more + 1) / (int(permutations) + 1)
    return {"corr": float(obs), "p_shift": float(p), "n": float(n)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", default="thesis/synthesis/output_coherence_beats_volatility/merged.csv")
    p.add_argument("--out", default="thesis/synthesis/output_subera_volatility_convergence")
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=909)
    args = p.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    m = pd.read_csv(args.merged).sort_values("time_bin", ascending=False).reset_index(drop=True)

    eras = [
        ("Cenozoic", 0.0, 66.0),
        ("Mesozoic", 66.0, 252.0),
        ("Paleozoic", 252.0, 600.0),
    ]

    results = {}
    for name, lo, hi in eras:
        sub = m[(m["time_bin"] >= lo) & (m["time_bin"] < hi)].copy().sort_values("time_bin", ascending=False)
        y = sub["functional_excess_similarity_js"].to_numpy(dtype=float)
        vol = sub["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
        coh = sub["delta_from_prev_T_effective_rank"].to_numpy(dtype=float)

        ctrl = np.column_stack(
            [
                sub["time_bin"].to_numpy(dtype=float),
                np.log1p(sub["n_localities"].to_numpy(dtype=float)),
                np.log1p(sub["marine_n_collections"].to_numpy(dtype=float)),
                np.log1p(sub["marine_n_occurrences"].to_numpy(dtype=float)),
                sub["provinciality"].to_numpy(dtype=float),
            ]
        )

        ry = _residualize(y, ctrl)
        rvol = _residualize(vol, ctrl)
        rcoh = _residualize(coh, ctrl)

        results[name] = {
            "n_bins": int(len(sub)),
            "volatility": _circular_shift_p(rvol, ry, permutations=int(args.permutations), seed=int(args.seed) + 1),
            "effective_rank": _circular_shift_p(rcoh, ry, permutations=int(args.permutations), seed=int(args.seed) + 2),
        }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# Sub-era robustness: volatility/coherence vs convergence (circular-shift null)",
        "",
        f"- input: `{Path(args.merged)}`",
        "",
        "Controls (within sub-era): time + sampling proxies + provinciality.",
        "",
    ]
    for era, stats in results.items():
        v = stats["volatility"]
        c = stats["effective_rank"]
        lines.extend(
            [
                f"## {era}",
                "",
                f"- bins: {int(stats['n_bins'])}",
                f"- volatility: corr={v['corr']:.3f}, shift_p={v['p_shift']:.3g}",
                f"- effective_rank: corr={c['corr']:.3f}, shift_p={c['p_shift']:.3g}",
                "",
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()

