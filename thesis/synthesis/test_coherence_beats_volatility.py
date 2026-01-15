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


def _plot_scatter(d: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    df = d[[x, y]].dropna().copy()
    if len(df) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.scatter(df[x], df[y], s=40, alpha=0.75, color="#1f77b4", edgecolors="none")
    A = np.vstack([df[x].to_numpy(), np.ones(len(df))]).T
    coef, *_ = np.linalg.lstsq(A, df[y].to_numpy(), rcond=None)
    xx = np.linspace(float(df[x].min()), float(df[x].max()), 60)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="black", linewidth=1.2, alpha=0.75)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


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


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(np.sum(mask)) < (X.shape[1] + 3):
        return float("nan")
    yy = y[mask]
    XX = X[mask]
    A = np.column_stack([np.ones(len(XX)), XX])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    pred = A.dot(beta)
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--convergence", default="thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv")
    p.add_argument("--pbdb-extended", default="data/processed/pbdb_occurrences_extended.parquet")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--out", default="thesis/synthesis/output_coherence_beats_volatility")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=131)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    conv = pd.read_csv(args.convergence)
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})
    pb = pd.read_parquet(
        args.pbdb_extended,
        columns=["occurrence_no", "collection_no", "reference_no", "mid_ma", "environment"],
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

    samp = (
        _agg(pb, prefix="")
        .merge(_agg(pb[pb["env_class"] == "marine"], prefix="marine_"), on="time_bin", how="left")
        .merge(_agg(pb[pb["env_class"] == "terrestrial"], prefix="terr_"), on="time_bin", how="left")
    )

    merged = conv.merge(earth, on="time_bin", how="left").merge(samp, on="time_bin", how="left")
    merged = merged.sort_values("time_bin", ascending=False).reset_index(drop=True)
    merged.to_csv(out_dir / "merged.csv", index=False)

    # Primary variables.
    y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
    vol = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)

    coh_ratio = merged["delta_from_prev_T_coherence_ratio"].to_numpy(dtype=float)
    coh_sign = merged["delta_from_prev_T_sign_agreement_frac"].to_numpy(dtype=float)
    patch_edges = merged["delta_from_prev_T_sign_edge_count"].to_numpy(dtype=float)
    patch_components = merged["delta_from_prev_T_sign_components"].to_numpy(dtype=float)
    pc1 = merged["delta_from_prev_T_pc1_frac"].to_numpy(dtype=float)
    eff_rank = merged["delta_from_prev_T_effective_rank"].to_numpy(dtype=float)

    t = merged["time_bin"].to_numpy(dtype=float)
    nloc = np.log1p(merged["n_localities"].to_numpy(dtype=float))
    ncoll = np.log1p(merged["marine_n_collections"].to_numpy(dtype=float))
    nocc = np.log1p(merged["marine_n_occurrences"].to_numpy(dtype=float))
    prov = merged["provinciality"].to_numpy(dtype=float)

    # Controls
    ctrl_time = np.column_stack([t])
    ctrl_time_loc = np.column_stack([t, nloc])
    ctrl_time_loc_samp = np.column_stack([t, nloc, ncoll, nocc])
    ctrl_time_loc_samp_prov = np.column_stack([t, nloc, ncoll, nocc, prov])

    results: dict[str, Any] = {"n_bins_total": int(len(merged))}

    def _run_block(name: str, x: np.ndarray, controls: np.ndarray, seed_off: int) -> None:
        results[name] = _partial_test(x=x, y=y, controls=controls, permutations=int(args.permutations), seed=int(args.seed) + seed_off)

    # Coherence vs convergence (increasing coherence should increase convergence).
    _run_block("coh_ratio__control_time", coh_ratio, ctrl_time, 10)
    _run_block("coh_ratio__control_time_loc", coh_ratio, ctrl_time_loc, 11)
    _run_block("coh_ratio__control_time_loc_samp", coh_ratio, ctrl_time_loc_samp, 12)
    _run_block("coh_ratio__control_time_loc_samp_prov", coh_ratio, ctrl_time_loc_samp_prov, 13)

    _run_block("coh_sign__control_time", coh_sign, ctrl_time, 20)
    _run_block("coh_sign__control_time_loc", coh_sign, ctrl_time_loc, 21)
    _run_block("coh_sign__control_time_loc_samp", coh_sign, ctrl_time_loc_samp, 22)
    _run_block("coh_sign__control_time_loc_samp_prov", coh_sign, ctrl_time_loc_samp_prov, 23)

    # Patchiness should go the other way (more patchy forcing -> less synchronized filtering).
    _run_block("patch_edges__control_time", patch_edges, ctrl_time, 30)
    _run_block("patch_edges__control_time_loc_samp_prov", patch_edges, ctrl_time_loc_samp_prov, 31)
    _run_block("patch_components__control_time", patch_components, ctrl_time, 32)
    _run_block("patch_components__control_time_loc_samp_prov", patch_components, ctrl_time_loc_samp_prov, 33)

    _run_block("pc1_frac__control_time", pc1, ctrl_time, 40)
    _run_block("pc1_frac__control_time_loc_samp_prov", pc1, ctrl_time_loc_samp_prov, 41)
    _run_block("effective_rank__control_time", eff_rank, ctrl_time, 42)
    _run_block("effective_rank__control_time_loc_samp_prov", eff_rank, ctrl_time_loc_samp_prov, 43)

    # Compare: coherence vs volatility magnitude (who "wins" when both included?).
    ctrl_base = ctrl_time_loc_samp_prov
    results["corr_vol_vs_coh_ratio"] = float(_corr(vol, coh_ratio))
    results["corr_vol_vs_coh_sign"] = float(_corr(vol, coh_sign))
    results["r2_base_time_loc_samp_prov"] = float(_ols_r2(y, ctrl_base))
    results["r2_plus_vol"] = float(_ols_r2(y, np.column_stack([ctrl_base, vol])))
    results["r2_plus_coh_ratio"] = float(_ols_r2(y, np.column_stack([ctrl_base, coh_ratio])))
    results["r2_plus_both_vol_coh_ratio"] = float(_ols_r2(y, np.column_stack([ctrl_base, vol, coh_ratio])))

    # Partial tests with both knobs (coherence controlling for magnitude, and vice versa).
    _run_block("coh_ratio__control_time_loc_samp_prov_plus_vol", coh_ratio, np.column_stack([ctrl_base, vol]), 60)
    _run_block("vol__control_time_loc_samp_prov_plus_coh_ratio", vol, np.column_stack([ctrl_base, coh_ratio]), 61)

    # Simple scatter for inspection.
    merged["y_resid_base"] = _residualize(y, ctrl_base)
    merged["vol_resid_base"] = _residualize(vol, ctrl_base)
    merged["coh_ratio_resid_base"] = _residualize(coh_ratio, ctrl_base)
    _plot_scatter(
        merged,
        x="vol_resid_base",
        y="y_resid_base",
        out_path=fig_dir / "volatility_vs_convergence_resid.png",
        title="Convergence residual vs volatility residual (controls: time+sampling+prov)",
    )
    _plot_scatter(
        merged,
        x="coh_ratio_resid_base",
        y="y_resid_base",
        out_path=fig_dir / "coherence_ratio_vs_convergence_resid.png",
        title="Convergence residual vs coherence residual (controls: time+sampling+prov)",
    )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Human-readable summary.
    def _fmt(name: str) -> str:
        r = results.get(name, {})
        if not isinstance(r, dict):
            return f"- {name}: (missing)"
        return (
            f"- {name}: corr={r.get('corr', float('nan')):.3f}, "
            f"iid_p={r.get('p_perm', float('nan')):.3g}, shift_p={r.get('p_shift', float('nan')):.3g}, n={int(r.get('n', 0))}"
        )

    lines = [
        "# Coherence beats magnitude? (first pass, full PBDB)",
        "",
        "Outcome: `functional_excess_similarity_js` (marine functional convergence beyond taxonomy).",
        "Magnitude: `delta_from_prev_T_field_meanabs` (mean |ΔT| field, 10 Myr step).",
        "Coherence: `delta_from_prev_T_coherence_ratio` = `|Δ global mean T| / mean(|ΔT field|)` (≈1 means mostly same-sign changes).",
        "",
        f"- bins used: {int(len(merged))}",
        f"- corr(vol, coh_ratio): {results['corr_vol_vs_coh_ratio']:.3f}",
        f"- R2 base (time+sampling+prov): {results['r2_base_time_loc_samp_prov']:.3f}",
        f"- R2 + volatility: {results['r2_plus_vol']:.3f}",
        f"- R2 + coherence_ratio: {results['r2_plus_coh_ratio']:.3f}",
        f"- R2 + both: {results['r2_plus_both_vol_coh_ratio']:.3f}",
        "",
        "## Partial correlation tests (controls noted in name; iid + circular-shift p-values)",
        "",
        _fmt("coh_ratio__control_time"),
        _fmt("coh_ratio__control_time_loc"),
        _fmt("coh_ratio__control_time_loc_samp"),
        _fmt("coh_ratio__control_time_loc_samp_prov"),
        "",
        _fmt("coh_ratio__control_time_loc_samp_prov_plus_vol"),
        _fmt("vol__control_time_loc_samp_prov_plus_coh_ratio"),
        "",
        "## Additional coherence/patchiness metrics (sanity checks)",
        "",
        _fmt("coh_sign__control_time_loc_samp_prov"),
        _fmt("patch_edges__control_time_loc_samp_prov"),
        _fmt("pc1_frac__control_time_loc_samp_prov"),
        _fmt("effective_rank__control_time_loc_samp_prov"),
        "",
        "## Outputs",
        "",
        f"- merged: `{out_dir / 'merged.csv'}`",
        f"- results: `{out_dir / 'analysis_results.json'}`",
        f"- figures: `{fig_dir}`",
        "",
        "Notes:",
        "- These are bin-level tests; publication-grade inference should use explicit time-series or hierarchical models.",
        "- Coherence metrics are derived from global ΔT fields (Li et al. 2022 CESM snapshots) and are sensitive to the chosen ΔT threshold.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()

