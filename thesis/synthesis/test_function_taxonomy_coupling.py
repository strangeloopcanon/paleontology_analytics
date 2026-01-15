from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _partial_test(
    *,
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    rx = _residualize(x, controls)
    ry = _residualize(y, controls)
    out = {}
    out.update(_iid_perm_p(rx, ry, permutations=int(permutations), seed=int(seed)))
    out.update(_circular_shift_p(rx, ry, permutations=int(permutations), seed=int(seed) + 1000))
    return out


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--convergence", default="thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv")
    p.add_argument("--pbdb-extended", default="data/processed/pbdb_occurrences_extended.parquet")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--out", default="thesis/synthesis/output_function_taxonomy_coupling")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=211)
    args = p.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    conv = pd.read_csv(args.convergence)
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})
    pb = pd.read_parquet(
        args.pbdb_extended,
        columns=["occurrence_no", "collection_no", "reference_no", "mid_ma", "environment"],
    )

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

    samp = (
        _agg(pb, prefix="")
        .merge(_agg(pb[pb["env_class"] == "marine"], prefix="marine_"), on="time_bin", how="left")
        .merge(_agg(pb[pb["env_class"] == "terrestrial"], prefix="terr_"), on="time_bin", how="left")
    )

    merged = conv.merge(earth, on="time_bin", how="left").merge(samp, on="time_bin", how="left")
    merged = merged.sort_values("time_bin", ascending=False).reset_index(drop=True)
    merged.to_csv(out_dir / "merged.csv", index=False)

    # Dependent variables: coupling slope in functional~taxonomic model per bin.
    # Prefer per-bin fits if available; fallback to the global-fit slope.
    js_col = "bin_js_slope_tax_to_func" if "bin_js_slope_tax_to_func" in merged.columns else "model_js_slope_tax_to_func"
    roles_col = (
        "bin_roles_slope_tax_to_func"
        if "bin_roles_slope_tax_to_func" in merged.columns
        else "model_roles_slope_tax_to_func"
    )
    y_js_slope = merged[js_col].to_numpy(dtype=float)
    y_roles_slope = merged[roles_col].to_numpy(dtype=float)

    js_int_col = "bin_js_intercept" if "bin_js_intercept" in merged.columns else "model_js_intercept"
    roles_int_col = "bin_roles_intercept" if "bin_roles_intercept" in merged.columns else "model_roles_intercept"
    y_js_int = merged[js_int_col].to_numpy(dtype=float)
    y_roles_int = merged[roles_int_col].to_numpy(dtype=float)

    # Predictors: magnitude + coherence.
    vol = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    coh_sign = merged["delta_from_prev_T_sign_agreement_frac"].to_numpy(dtype=float)
    eff_rank = merged["delta_from_prev_T_effective_rank"].to_numpy(dtype=float)

    # Controls: time + sampling proxies + province structure.
    t = merged["time_bin"].to_numpy(dtype=float)
    nloc = np.log1p(merged["n_localities"].to_numpy(dtype=float))
    ncoll = np.log1p(merged["marine_n_collections"].to_numpy(dtype=float))
    nocc = np.log1p(merged["marine_n_occurrences"].to_numpy(dtype=float))
    prov = merged["provinciality"].to_numpy(dtype=float)
    ctrl = np.column_stack([t, nloc, ncoll, nocc, prov])

    results: dict[str, Any] = {}
    def _run(name: str, x: np.ndarray, y: np.ndarray, seed_off: int) -> None:
        results[name] = _partial_test(x=x, y=y, controls=ctrl, permutations=int(args.permutations), seed=int(args.seed) + seed_off)

    for yname, y in [
        ("js_slope", y_js_slope),
        ("roles_slope", y_roles_slope),
        ("js_intercept", y_js_int),
        ("roles_intercept", y_roles_int),
    ]:
        _run(f"{yname}__vs_vol", vol, y, 10)
        _run(f"{yname}__vs_coh_sign", coh_sign, y, 20)
        _run(f"{yname}__vs_eff_rank", eff_rank, y, 30)

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    def _fmt(key: str) -> str:
        r = results.get(key, {})
        if not isinstance(r, dict):
            return f"- {key}: (missing)"
        return (
            f"- {key}: corr={r.get('corr', float('nan')):.3f}, "
            f"iid_p={r.get('p_perm', float('nan')):.3g}, shift_p={r.get('p_shift', float('nan')):.3g}, n={int(r.get('n', 0))}"
        )

    lines = [
        "# Functional↔taxonomic coupling under forcing (first pass)",
        "",
        "Each bin fits `functional_similarity ~ taxonomic_similarity` across province pairs; we test whether the fitted slope changes with forcing.",
        "",
        "Controls: time + sampling proxies (localities/collections/occurrences) + provinciality; iid + circular-shift p-values.",
        "",
        "## Results",
        "",
        _fmt("js_slope__vs_vol"),
        _fmt("js_slope__vs_coh_sign"),
        _fmt("js_slope__vs_eff_rank"),
        "",
        _fmt("roles_slope__vs_vol"),
        _fmt("roles_slope__vs_coh_sign"),
        _fmt("roles_slope__vs_eff_rank"),
        "",
        _fmt("js_intercept__vs_vol"),
        _fmt("js_intercept__vs_coh_sign"),
        _fmt("js_intercept__vs_eff_rank"),
        "",
        _fmt("roles_intercept__vs_vol"),
        _fmt("roles_intercept__vs_coh_sign"),
        _fmt("roles_intercept__vs_eff_rank"),
        "",
        "Interpretation:",
        "- Negative corr(slope, forcing) implies functional similarity becomes less dependent on taxonomic similarity (more decoupling).",
        "- Positive corr(slope, forcing) implies tighter coupling (functions track taxa more).",
        "- Positive corr(intercept, forcing) implies higher baseline functional similarity at a given (low) taxonomic similarity (a decoupling signature).",
        "",
        "## Outputs",
        "",
        f"- merged: `{out_dir / 'merged.csv'}`",
        f"- results: `{out_dir / 'analysis_results.json'}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
