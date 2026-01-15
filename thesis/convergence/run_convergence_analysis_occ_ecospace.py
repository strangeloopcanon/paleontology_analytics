from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon


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


def _analysis_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer paleo coords with fallback; normalize longitude to [-180, 180].
    out = df.copy()
    out["analysis_lat"] = out["paleolat"].where(out["paleolat"].notna(), out["lat"])
    out["analysis_lng"] = out["paleolng"].where(out["paleolng"].notna(), out["lng"])
    out = out.dropna(subset=["analysis_lat", "analysis_lng"]).copy()
    out["analysis_lat"] = pd.to_numeric(out["analysis_lat"], errors="coerce").clip(-90, 90)
    out["analysis_lng"] = pd.to_numeric(out["analysis_lng"], errors="coerce")
    out["analysis_lng"] = out["analysis_lng"].where(out["analysis_lng"] <= 180, out["analysis_lng"] - 360)
    out["analysis_lng"] = out["analysis_lng"].where(out["analysis_lng"] >= -180, out["analysis_lng"] + 360)
    out = out.dropna(subset=["analysis_lat", "analysis_lng"]).copy()
    return out


def _js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    # SciPy returns JS distance in [0, 1] when base=2 (default).
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p.astype(float) / float(p.sum())
    q = q.astype(float) / float(q.sum())
    d = float(jensenshannon(p, q))
    if not np.isfinite(d):
        return float("nan")
    return float(1.0 - np.clip(d, 0.0, 1.0))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter / union) if union else float("nan")


def _fit_ols_1d(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    A = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
    return {"intercept": intercept, "slope": slope, "r2": r2}


@dataclass(frozen=True)
class OccEcospaceCols:
    taxon_env: str
    diet: str
    motility: str
    life_habit: str


def _infer_ecospace_cols(df: pd.DataFrame) -> OccEcospaceCols:
    # PBDB CSV exports use friendly column names.
    for required in ["taxon_environment", "diet", "motility", "life_habit"]:
        if required not in df.columns:
            raise ValueError(f"PBDB occs CSV missing ecospace column: {required}")
    return OccEcospaceCols(
        taxon_env="taxon_environment",
        diet="diet",
        motility="motility",
        life_habit="life_habit",
    )


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y, "n_localities"]].dropna().copy()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.3, 4.9))
    ax.scatter(d[x], d[y], s=np.clip(d["n_localities"] * 2.0, 20, 200), alpha=0.75, color="#1f77b4", edgecolors="none")
    A = np.vstack([d[x].to_numpy(), np.ones(len(d))]).T
    coef, _, _, _ = np.linalg.lstsq(A, d[y].to_numpy(), rcond=None)
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
    p.add_argument("--pbdb-csv", default="data/raw/pbdb_occurrences_mammalia_ecospace_paged.csv")
    p.add_argument("--out", default="thesis/convergence/output_occ_ecospace_mammalia")
    p.add_argument("--env-substr", default="terrestrial", help="Substring match on PBDB ecospace `taxon_environment`.")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--grid-deg", type=float, default=10.0)
    p.add_argument("--min-genera-per-region", type=int, default=25)
    p.add_argument("--min-pairs-per-bin", type=int, default=200)
    p.add_argument("--max-pairs-per-bin", type=int, default=30000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    df = pd.read_csv(args.pbdb_csv, low_memory=False)
    cols = _infer_ecospace_cols(df)

    # Basic cleaning.
    df["genus"] = df["genus"].map(_clean_name)
    df = df.dropna(subset=["genus", "max_ma", "min_ma"]).copy()
    df["max_ma"] = pd.to_numeric(df["max_ma"], errors="coerce")
    df["min_ma"] = pd.to_numeric(df["min_ma"], errors="coerce")
    df = df.dropna(subset=["max_ma", "min_ma"]).copy()
    df["mid_ma"] = (df["max_ma"] + df["min_ma"]) / 2.0

    # Ecospace fields.
    for c in [cols.taxon_env, cols.diet, cols.motility, cols.life_habit]:
        df[c] = df[c].map(_clean_name)
    df = df.dropna(subset=[cols.diet, cols.motility, cols.life_habit]).copy()

    # Environment filter.
    env_sub = str(args.env_substr).strip().lower()
    if env_sub:
        df = df[df[cols.taxon_env].astype(str).str.lower().str.contains(env_sub, na=False)].copy()

    df = _analysis_lat_lng(df)
    df["time_bin"] = (pd.to_numeric(df["mid_ma"], errors="coerce") / float(args.time_bin_myr)).round() * float(
        args.time_bin_myr
    )
    df["lat_bin"] = (df["analysis_lat"] / float(args.grid_deg)).round() * float(args.grid_deg)
    df["lng_bin"] = (df["analysis_lng"] / float(args.grid_deg)).round() * float(args.grid_deg)
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))

    df["role_id"] = df[cols.diet].astype(str) + "|" + df[cols.motility].astype(str) + "|" + df[cols.life_habit].astype(str)

    # De-duplicate within locality×bin by genus to reduce oversampling.
    df = df.drop_duplicates(subset=["time_bin", "locality", "genus"]).copy()

    # Build per locality: genus set and role counts.
    genus_sets = (
        df.groupby(["time_bin", "locality"])["genus"]
        .agg(lambda s: set(s.astype(str)))
        .rename("genus_set")
        .reset_index()
    )
    genus_counts = df.groupby(["time_bin", "locality"])["genus"].nunique().rename("n_genera").reset_index()
    genus_sets = genus_sets.merge(genus_counts, on=["time_bin", "locality"], how="left")

    role_counts = (
        df.groupby(["time_bin", "locality", "role_id"])["genus"]
        .nunique()
        .rename("n_genera_role")
        .reset_index()
    )

    all_roles = sorted(role_counts["role_id"].unique())
    role_index = {r: i for i, r in enumerate(all_roles)}

    # Precompute per locality vectors.
    vectors: dict[tuple[float, tuple[float, float]], np.ndarray] = {}
    for (tb, loc), sub in role_counts.groupby(["time_bin", "locality"], sort=False):
        v = np.zeros(len(all_roles), dtype=float)
        for _, row in sub.iterrows():
            rid = row["role_id"]
            v[role_index[rid]] = float(row["n_genera_role"])
        vectors[(float(tb), tuple(loc))] = v

    bins = sorted(genus_sets["time_bin"].unique(), reverse=True)
    metrics_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))

    for tb in bins:
        tb = float(tb)
        loc_df = genus_sets[genus_sets["time_bin"] == tb].copy()
        loc_df = loc_df[loc_df["n_genera"] >= int(args.min_genera_per_region)].copy()
        localities = list(loc_df["locality"])
        if len(localities) < 6:
            continue

        loc_sets = {tuple(loc): loc_df.loc[loc_df["locality"] == loc, "genus_set"].iloc[0] for loc in localities}
        loc_vecs = {tuple(loc): vectors[(tb, tuple(loc))] for loc in localities if (tb, tuple(loc)) in vectors}

        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for i in range(len(localities) - 1):
            for j in range(i + 1, len(localities)):
                pairs.append((tuple(localities[i]), tuple(localities[j])))

        if len(pairs) > int(args.max_pairs_per_bin):
            idx = rng.choice(len(pairs), size=int(args.max_pairs_per_bin), replace=False)
            pairs = [pairs[int(j)] for j in idx]

        func_sims = []
        tax_sims = []
        for loc_a, loc_b in pairs:
            va = loc_vecs.get(loc_a)
            vb = loc_vecs.get(loc_b)
            if va is None or vb is None:
                continue
            func = _js_similarity(va, vb)
            tax = _jaccard_similarity(loc_sets.get(loc_a, set()), loc_sets.get(loc_b, set()))
            if not np.isfinite(func) or not np.isfinite(tax):
                continue
            func_sims.append(func)
            tax_sims.append(tax)
            pair_rows.append(
                {
                    "time_bin": tb,
                    "loc_a": str(loc_a),
                    "loc_b": str(loc_b),
                    "functional_similarity_js": float(func),
                    "taxonomic_similarity": float(tax),
                }
            )

        if len(func_sims) < int(args.min_pairs_per_bin):
            continue

        mean_func = float(np.mean(func_sims))
        mean_tax = float(np.mean(tax_sims))
        metrics_rows.append(
            {
                "time_bin": tb,
                "n_localities": int(len(localities)),
                "n_pairs": int(len(func_sims)),
                "mean_functional_similarity_js": mean_func,
                "mean_taxonomic_similarity": mean_tax,
                "provinciality": float(1.0 - mean_tax),
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    metrics = pd.DataFrame(metrics_rows).sort_values("time_bin", ascending=False).reset_index(drop=True)

    if len(pairwise) == 0 or len(metrics) == 0:
        raise RuntimeError("No pairwise data or bins; try lowering thresholds.")

    # Global relationship: functional similarity ~ taxonomic similarity.
    x = pairwise["taxonomic_similarity"].to_numpy(dtype=float)
    fit_js = _fit_ols_1d(x, pairwise["functional_similarity_js"].to_numpy(dtype=float))
    pairwise["functional_js_pred"] = fit_js["intercept"] + fit_js["slope"] * x
    pairwise["functional_js_residual"] = pairwise["functional_similarity_js"] - pairwise["functional_js_pred"]

    # Per-bin coupling fits (optional mechanistic diagnostics).
    bin_rows = []
    for tb, g in pairwise.groupby("time_bin", sort=False):
        f_js = _fit_ols_1d(
            g["taxonomic_similarity"].to_numpy(dtype=float),
            g["functional_similarity_js"].to_numpy(dtype=float),
        )
        bin_rows.append(
            {
                "time_bin": float(tb),
                "bin_js_intercept": float(f_js["intercept"]),
                "bin_js_slope_tax_to_func": float(f_js["slope"]),
                "bin_js_r2": float(f_js["r2"]),
            }
        )
    bin_fit = pd.DataFrame(bin_rows)

    res = pairwise.groupby("time_bin")[["functional_js_residual"]].mean().reset_index()
    res = res.rename(columns={"functional_js_residual": "functional_excess_similarity_js"})

    metrics = metrics.merge(res, on="time_bin", how="left").merge(bin_fit, on="time_bin", how="left")
    metrics["model_js_intercept"] = float(fit_js["intercept"])
    metrics["model_js_slope_tax_to_func"] = float(fit_js["slope"])
    metrics["model_js_r2"] = float(fit_js["r2"])

    metrics.to_csv(out_dir / "timebin_metrics.csv", index=False)
    pairwise.to_csv(out_dir / "pairwise_sample.csv", index=False)
    (out_dir / "analysis_results.json").write_text(json.dumps({"global_fit": fit_js, "n_bins": int(len(metrics))}, indent=2) + "\n")

    _plot_scatter(
        metrics,
        x="provinciality",
        y="functional_excess_similarity_js",
        out_path=fig_dir / "scatter_provinciality_vs_excess_similarity.png",
        title="Provinciality vs functional excess similarity",
    )

    lines = [
        "# Convergence (occurrence-level ecospace): PBDB subset",
        "",
        f"- Input PBDB CSV: `{Path(args.pbdb_csv)}`",
        f"- Taxon-environment filter: `{args.env_substr}` (substring on `taxon_environment`)",
        f"- Bins written: {len(metrics)}",
        f"- Global fit R²: {fit_js['r2']:.3f}",
        "",
        "## Outputs",
        "",
        f"- time bins: `{out_dir / 'timebin_metrics.csv'}`",
        f"- pairwise sample: `{out_dir / 'pairwise_sample.csv'}`",
        f"- meta: `{out_dir / 'analysis_results.json'}`",
        f"- figures: `{fig_dir}`",
        "",
        "## Notes",
        "",
        "- This mirrors the marine pipeline but uses PBDB *occurrence-level* ecospace fields (`diet`, `motility`, `life_habit`) from `occs/list`.",
        "- For forcing tests, merge `timebin_metrics.csv` with `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv` on `time_bin`.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
