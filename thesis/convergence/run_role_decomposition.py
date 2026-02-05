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
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _analysis_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
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


def _entropy(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    s = float(np.sum(counts))
    if s <= 0:
        return float("nan")
    p = counts / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


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


def _coarse_diet(x: str | None) -> str | None:
    s = _clean_name(x)
    if s is None:
        return None
    t = s.lower()
    if "suspension" in t:
        return "suspension feeder"
    if "deposit" in t:
        return "deposit feeder"
    if "carnivore" in t:
        return "carnivore"
    if "detritivore" in t:
        return "detritivore"
    if "herbivore" in t:
        return "herbivore"
    if "grazer" in t:
        return "grazer"
    if "photosymbiotic" in t:
        return "photosymbiotic"
    return s


def _coarse_motility(x: str | None) -> str | None:
    s = _clean_name(x)
    if s is None:
        return None
    t = s.lower()
    if "actively mobile" in t:
        return "actively mobile"
    if "facultatively mobile" in t:
        return "facultatively mobile"
    if "stationary" in t:
        return "stationary"
    return s


def _coarse_habit(x: str | None) -> str | None:
    s = _clean_name(x)
    if s is None:
        return None
    t = s.lower()
    if "nekton" in t:
        return "nektonic"
    if "plankt" in t:
        return "planktic"
    if "semi-infaunal" in t:
        return "semi-infaunal"
    if "infaunal" in t:
        return "infaunal"
    if "epifaunal" in t:
        return "epifaunal"
    return s


@dataclass(frozen=True)
class LocalityVectors:
    genus_set: set[str]
    role_vec: np.ndarray
    diet_vec: np.ndarray
    motility_vec: np.ndarray
    habit_vec: np.ndarray


def _vectorize_counts(counts: pd.DataFrame, categories: list[str], *, cat_col: str, value_col: str) -> np.ndarray:
    idx = {c: i for i, c in enumerate(categories)}
    v = np.zeros(len(categories), dtype=float)
    for _, row in counts.iterrows():
        cat = row[cat_col]
        if cat not in idx:
            continue
        v[idx[cat]] = float(row[value_col])
    return v


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y]].dropna()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
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
    p.add_argument("--out", default="thesis/convergence/output_role_decomposition")
    p.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--mapping", default="thesis/convergence/output_v2/ecospace_genus_mapping.csv")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--env", default="marine", choices=["marine", "terrestrial"])
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--grid-deg", type=float, default=10.0)
    p.add_argument("--min-genera-per-region", type=int, default=25)
    p.add_argument("--max-pairs-per-bin", type=int, default=30000)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # Load PBDB occurrences from repo.
    cols = [
        "source_db",
        "occurrence_id",
        "mid_ma",
        "lat",
        "lng",
        "paleolat",
        "paleolng",
        "genus",
    ]
    occ = pd.read_parquet(args.pbdb, columns=cols)
    occ = occ[occ["source_db"] == "PBDB"].drop_duplicates(subset=["source_db", "occurrence_id"]).copy()
    occ["genus"] = occ["genus"].map(_clean_name)
    occ = occ.dropna(subset=["genus", "mid_ma"]).copy()

    mapping = pd.read_csv(args.mapping)
    mapping["genus"] = mapping["genus"].map(_clean_name)
    mapping = mapping.dropna(subset=["genus"]).copy()
    mapping["role_id"] = mapping["role_id"].map(_clean_name)
    mapping["jev"] = mapping["jev"].map(_clean_name)
    mapping["diet_coarse"] = mapping["jdt"].map(_coarse_diet)
    mapping["motility_coarse"] = mapping["jmo"].map(_coarse_motility)
    mapping["habit_coarse"] = mapping["jlh"].map(_coarse_habit)

    df = occ.merge(mapping, on="genus", how="left")

    # Focus on environment (PBDB ecospace-coded) and require full role_id (for consistent projections).
    df = df[df["jev"].astype(str).str.contains(str(args.env), case=False, na=False)].copy()
    df = df.dropna(subset=["role_id", "diet_coarse", "motility_coarse", "habit_coarse"]).copy()

    df = _analysis_lat_lng(df)
    df["time_bin"] = (pd.to_numeric(df["mid_ma"], errors="coerce") / float(args.time_bin_myr)).round() * float(
        args.time_bin_myr
    )
    df["lat_bin"] = (df["analysis_lat"] / float(args.grid_deg)).round() * float(args.grid_deg)
    df["lng_bin"] = (df["analysis_lng"] / float(args.grid_deg)).round() * float(args.grid_deg)
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))

    # De-duplicate within locality×bin by genus to reduce oversampling.
    df = df.drop_duplicates(subset=["time_bin", "locality", "genus"]).copy()

    genus_sets = (
        df.groupby(["time_bin", "locality"])["genus"]
        .agg(lambda s: set(s.astype(str)))
        .rename("genus_set")
        .reset_index()
    )
    genus_counts = df.groupby(["time_bin", "locality"])["genus"].nunique().rename("n_genera").reset_index()
    genus_sets = genus_sets.merge(genus_counts, on=["time_bin", "locality"], how="left")

    # Role / diet / motility / habit counts per locality×bin (unique genera).
    role_counts = (
        df.groupby(["time_bin", "locality", "role_id"])["genus"]
        .nunique()
        .rename("n_genera_role")
        .reset_index()
    )
    diet_counts = (
        df.groupby(["time_bin", "locality", "diet_coarse"])["genus"]
        .nunique()
        .rename("n_genera_diet")
        .reset_index()
    )
    motility_counts = (
        df.groupby(["time_bin", "locality", "motility_coarse"])["genus"]
        .nunique()
        .rename("n_genera_motility")
        .reset_index()
    )
    habit_counts = (
        df.groupby(["time_bin", "locality", "habit_coarse"])["genus"]
        .nunique()
        .rename("n_genera_habit")
        .reset_index()
    )

    all_roles = sorted(role_counts["role_id"].unique())
    all_diets = sorted(diet_counts["diet_coarse"].unique())
    all_mot = sorted(motility_counts["motility_coarse"].unique())
    all_hab = sorted(habit_counts["habit_coarse"].unique())

    # Precompute locality vectors.
    locality_map: dict[tuple[float, tuple[float, float]], LocalityVectors] = {}
    for (t, loc), sub in genus_sets.groupby(["time_bin", "locality"], sort=False):
        t = float(t)
        loc_t = tuple(loc)
        n_g = int(sub["n_genera"].iloc[0])
        if n_g < int(args.min_genera_per_region):
            continue

        genus_set = set(sub["genus_set"].iloc[0])
        rc = role_counts[(role_counts["time_bin"] == t) & (role_counts["locality"] == loc_t)]
        dc = diet_counts[(diet_counts["time_bin"] == t) & (diet_counts["locality"] == loc_t)]
        mc = motility_counts[(motility_counts["time_bin"] == t) & (motility_counts["locality"] == loc_t)]
        hc = habit_counts[(habit_counts["time_bin"] == t) & (habit_counts["locality"] == loc_t)]

        locality_map[(t, loc_t)] = LocalityVectors(
            genus_set=genus_set,
            role_vec=_vectorize_counts(rc, all_roles, cat_col="role_id", value_col="n_genera_role"),
            diet_vec=_vectorize_counts(dc, all_diets, cat_col="diet_coarse", value_col="n_genera_diet"),
            motility_vec=_vectorize_counts(mc, all_mot, cat_col="motility_coarse", value_col="n_genera_motility"),
            habit_vec=_vectorize_counts(hc, all_hab, cat_col="habit_coarse", value_col="n_genera_habit"),
        )

    bins = sorted({k[0] for k in locality_map.keys()}, reverse=True)
    if not bins:
        raise RuntimeError("No bins after filtering; try lowering --min-genera-per-region.")

    rng = np.random.default_rng(int(args.seed))
    pair_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []

    # Bin-level richness/entropy computed globally across all localities in the bin.
    def _bin_global_counts(sub: pd.DataFrame, cats: list[str], cat_col: str) -> np.ndarray:
        c = sub.groupby(cat_col)["genus"].nunique()
        v = np.zeros(len(cats), dtype=float)
        idx = {c: i for i, c in enumerate(cats)}
        for k, val in c.items():
            if k in idx:
                v[idx[k]] = float(val)
        return v

    for t in bins:
        localities = [loc for (tt, loc) in locality_map.keys() if float(tt) == float(t)]
        if len(localities) < 6:
            continue

        # Pair list (sample only if necessary).
        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for i in range(len(localities) - 1):
            for j in range(i + 1, len(localities)):
                pairs.append((tuple(localities[i]), tuple(localities[j])))
        if len(pairs) > int(args.max_pairs_per_bin):
            idx = rng.choice(len(pairs), size=int(args.max_pairs_per_bin), replace=False)
            pairs = [pairs[int(j)] for j in idx]

        for loc_a, loc_b in pairs:
            a = locality_map[(float(t), tuple(loc_a))]
            b = locality_map[(float(t), tuple(loc_b))]
            tax = _jaccard_similarity(a.genus_set, b.genus_set)
            if not np.isfinite(tax):
                continue
            pair_rows.append(
                {
                    "time_bin": float(t),
                    "loc_a": str(tuple(loc_a)),
                    "loc_b": str(tuple(loc_b)),
                    "taxonomic_similarity": float(tax),
                    "sim_role_js": float(_js_similarity(a.role_vec, b.role_vec)),
                    "sim_diet_js": float(_js_similarity(a.diet_vec, b.diet_vec)),
                    "sim_motility_js": float(_js_similarity(a.motility_vec, b.motility_vec)),
                    "sim_habit_js": float(_js_similarity(a.habit_vec, b.habit_vec)),
                }
            )

        # Global bin summaries.
        sub = df[df["time_bin"] == float(t)]
        role_global = _bin_global_counts(sub, all_roles, "role_id")
        diet_global = _bin_global_counts(sub, all_diets, "diet_coarse")
        mot_global = _bin_global_counts(sub, all_mot, "motility_coarse")
        hab_global = _bin_global_counts(sub, all_hab, "habit_coarse")
        bin_rows.append(
            {
                "time_bin": float(t),
                "n_localities": int(len(localities)),
                "n_roles": int(np.sum(role_global > 0)),
                "n_diets": int(np.sum(diet_global > 0)),
                "n_motilities": int(np.sum(mot_global > 0)),
                "n_habits": int(np.sum(hab_global > 0)),
                "entropy_roles": float(_entropy(role_global)),
                "entropy_diets": float(_entropy(diet_global)),
                "entropy_motility": float(_entropy(mot_global)),
                "entropy_habit": float(_entropy(hab_global)),
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    bins_df = pd.DataFrame(bin_rows).sort_values("time_bin", ascending=False).reset_index(drop=True)

    if len(pairwise) == 0 or len(bins_df) == 0:
        raise RuntimeError("No pairwise data or bins; try lowering thresholds.")

    # Fit global relationships: similarity ~ taxonomic similarity.
    x = pairwise["taxonomic_similarity"].to_numpy(dtype=float)
    fits: dict[str, dict[str, float]] = {}
    for col in ["sim_role_js", "sim_diet_js", "sim_motility_js", "sim_habit_js"]:
        fit = _fit_ols_1d(x, pairwise[col].to_numpy(dtype=float))
        fits[col] = fit
        pairwise[f"{col}_pred"] = fit["intercept"] + fit["slope"] * x
        pairwise[f"{col}_resid"] = pairwise[col] - pairwise[f"{col}_pred"]

    # Bin-level residual averages as “excess similarity”.
    resid_cols = [f"{c}_resid" for c in fits.keys()]
    residuals = pairwise.groupby("time_bin")[resid_cols].mean().reset_index()
    residuals = residuals.rename(
        columns={
            "sim_role_js_resid": "excess_role_js",
            "sim_diet_js_resid": "excess_diet_js",
            "sim_motility_js_resid": "excess_motility_js",
            "sim_habit_js_resid": "excess_habit_js",
        }
    )
    bins_df = bins_df.merge(residuals, on="time_bin", how="left")
    bins_df["model_role_r2"] = float(fits["sim_role_js"]["r2"])
    bins_df["model_diet_r2"] = float(fits["sim_diet_js"]["r2"])
    bins_df["model_motility_r2"] = float(fits["sim_motility_js"]["r2"])
    bins_df["model_habit_r2"] = float(fits["sim_habit_js"]["r2"])

    # Merge independent forcing series.
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})
    merged = bins_df.merge(earth, on="time_bin", how="left")
    merged.to_csv(out_dir / "timebin_metrics_decomposition.csv", index=False)
    pairwise.to_csv(out_dir / "pairwise_decomposition.csv", index=False)

    # Role ubiquity: which diet/motility/habit categories become more widespread in volatile vs stable bins?
    volatility_col = "delta_from_prev_T_field_meanabs"
    merged2 = merged.dropna(subset=[volatility_col]).copy()
    if len(merged2) >= 8:
        q_lo = float(merged2[volatility_col].quantile(0.25))
        q_hi = float(merged2[volatility_col].quantile(0.75))
    else:
        q_lo = float("nan")
        q_hi = float("nan")

    def _category_occupancy(table: pd.DataFrame, *, cat_col: str) -> pd.DataFrame:
        d = table.copy()
        d = d.dropna(subset=["time_bin", "locality", cat_col]).copy()
        d = d.drop_duplicates(subset=["time_bin", "locality", cat_col]).copy()
        loc_counts = d.groupby("time_bin")["locality"].nunique().rename("n_localities").reset_index()
        occ = d.groupby(["time_bin", cat_col])["locality"].nunique().rename("n_localities_with_cat").reset_index()
        occ = occ.merge(loc_counts, on="time_bin", how="left")
        occ["occupancy_frac"] = occ["n_localities_with_cat"] / occ["n_localities"]
        return occ

    occ_diet = _category_occupancy(df, cat_col="diet_coarse")
    occ_mot = _category_occupancy(df, cat_col="motility_coarse")
    occ_hab = _category_occupancy(df, cat_col="habit_coarse")
    occ_diet.to_csv(out_dir / "diet_occupancy_timeseries.csv", index=False)
    occ_mot.to_csv(out_dir / "motility_occupancy_timeseries.csv", index=False)
    occ_hab.to_csv(out_dir / "habit_occupancy_timeseries.csv", index=False)

    def _occupancy_partial_corrs(occ: pd.DataFrame, *, cat_col: str, min_bins: int = 12) -> pd.DataFrame:
        m = occ.merge(merged2[["time_bin", volatility_col]], on="time_bin", how="inner")
        cat_perms = int(min(5000, int(args.permutations)))
        rows: list[dict[str, Any]] = []
        for cat, sub in m.groupby(cat_col, sort=False):
            n_bins = int(sub["time_bin"].nunique())
            if n_bins < int(min_bins):
                continue
            x = sub[volatility_col].to_numpy(dtype=float)
            y = sub["occupancy_frac"].to_numpy(dtype=float)
            t = sub["time_bin"].to_numpy(dtype=float)
            pc = _partial_corr_perm(x, y, t, permutations=cat_perms, seed=int(args.seed) + 800)
            mask = np.isfinite(x) & np.isfinite(y)
            raw_corr = float(np.corrcoef(x[mask], y[mask])[0, 1]) if int(np.sum(mask)) >= 3 else float("nan")
            rows.append(
                {
                    cat_col: str(cat),
                    "n_bins": n_bins,
                    "partial_corr": float(pc.get("corr", float("nan"))),
                    "partial_p_perm": float(pc.get("p_perm", float("nan"))),
                    "raw_corr": raw_corr,
                }
            )
        out = pd.DataFrame(rows)
        if len(out) == 0:
            return out
        return out.sort_values("partial_corr", ascending=False).reset_index(drop=True)

    def _contrast_quartiles(occ: pd.DataFrame, *, cat_name: str) -> pd.DataFrame:
        m = occ.merge(merged2[["time_bin", volatility_col]], on="time_bin", how="inner")
        if not np.isfinite(q_lo) or not np.isfinite(q_hi):
            return pd.DataFrame()
        stable = m[m[volatility_col] <= q_lo]
        volatile = m[m[volatility_col] >= q_hi]
        if len(stable) == 0 or len(volatile) == 0:
            return pd.DataFrame()
        s = stable.groupby(cat_name)["occupancy_frac"].mean().rename("occupancy_stable_q25").reset_index()
        v = volatile.groupby(cat_name)["occupancy_frac"].mean().rename("occupancy_volatile_q75").reset_index()
        out = s.merge(v, on=cat_name, how="outer").fillna(0.0)
        out["delta_volatile_minus_stable"] = out["occupancy_volatile_q75"] - out["occupancy_stable_q25"]
        return out.sort_values("delta_volatile_minus_stable", ascending=False).reset_index(drop=True)

    diet_contrast = _contrast_quartiles(occ_diet, cat_name="diet_coarse")
    mot_contrast = _contrast_quartiles(occ_mot, cat_name="motility_coarse")
    hab_contrast = _contrast_quartiles(occ_hab, cat_name="habit_coarse")
    diet_pc = _occupancy_partial_corrs(occ_diet, cat_col="diet_coarse")
    mot_pc = _occupancy_partial_corrs(occ_mot, cat_col="motility_coarse")
    hab_pc = _occupancy_partial_corrs(occ_hab, cat_col="habit_coarse")

    if len(diet_contrast) > 0:
        diet_contrast.to_csv(out_dir / "diet_occupancy_contrast.csv", index=False)
    if len(mot_contrast) > 0:
        mot_contrast.to_csv(out_dir / "motility_occupancy_contrast.csv", index=False)
    if len(hab_contrast) > 0:
        hab_contrast.to_csv(out_dir / "habit_occupancy_contrast.csv", index=False)
    if len(diet_pc) > 0:
        diet_pc.to_csv(out_dir / "diet_occupancy_partialcorr.csv", index=False)
    if len(mot_pc) > 0:
        mot_pc.to_csv(out_dir / "motility_occupancy_partialcorr.csv", index=False)
    if len(hab_pc) > 0:
        hab_pc.to_csv(out_dir / "habit_occupancy_partialcorr.csv", index=False)

    # Correlation tests against independent forcing (with partial corr controlling for time).
    results: dict[str, Any] = {
        "n_bins": int(len(merged2)),
        "volatility_col": volatility_col,
        "quartiles": {"q25": q_lo, "q75": q_hi},
        "fits": fits,
    }
    y_cols = ["excess_role_js", "excess_diet_js", "excess_motility_js", "excess_habit_js", "entropy_roles"]
    for i, ycol in enumerate(y_cols):
        xvol = merged2[volatility_col].to_numpy(dtype=float)
        y = merged2[ycol].to_numpy(dtype=float)
        t = merged2["time_bin"].to_numpy(dtype=float)
        results[f"corr_{ycol}"] = _perm_test_corr(xvol, y, permutations=int(args.permutations), seed=int(args.seed) + i)
        results[f"partial_corr_{ycol}_control_time"] = _partial_corr_perm(
            xvol, y, t, permutations=int(args.permutations), seed=int(args.seed) + 100 + i
        )
        _plot_scatter(
            merged2,
            x=volatility_col,
            y=ycol,
            out_path=fig_dir / f"scatter_{volatility_col}__{ycol}.png",
            title=f"{ycol} vs {volatility_col}",
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Summary markdown.
    def _fmt_corr(key: str) -> str:
        r = results.get(key, {})
        return f"corr={float(r.get('corr')):.3f}, perm-p={float(r.get('p_perm')):.3g}, n={int(r.get('n') or 0)}"

    def _fmt_pcorr(key: str) -> str:
        r = results.get(key, {})
        return f"partial corr={float(r.get('corr')):.3f}, perm-p={float(r.get('p_perm')):.3g}"

    lines = [
        "# Role decomposition: what drives ecospace convergence?",
        "",
        "This extends the PBDB ecospace convergence pipeline by decomposing functional similarity into coarse ecospace axes:",
        "`diet`, `motility`, and `life habit` (plus the full role combination).",
        "",
        f"- Environment filter: `{str(args.env)}` (from PBDB ecospace field `jev`)",
        "",
        "Convergence metric: excess similarity = residual of (functional similarity ~ taxonomic similarity) across locality-pairs.",
        "",
        "## Key tests vs independent forcing (Li et al. 2022 CESM)",
        "",
        f"- Volatility series: `{args.earth}` column `{volatility_col}`",
        f"- Convergence decomposition metrics: `{out_dir / 'timebin_metrics_decomposition.csv'}`",
        "",
        "### Correlations (bin-level)",
        "",
    ]
    for ycol in y_cols:
        lines.append(f"- {ycol}: {_fmt_corr(f'corr_{ycol}')} ; {_fmt_pcorr(f'partial_corr_{ycol}_control_time')}")

    if np.isfinite(q_lo) and np.isfinite(q_hi):
        lines.extend(
            [
                "",
                "## Which categories get more widespread in volatile climates?",
                "",
                "We compare category occupancy across localities between the top and bottom volatility quartiles (by "
                f"`{volatility_col}`). Deltas are (volatile q75 – stable q25).",
                "",
            ]
        )
        if len(diet_contrast) > 0:
            top = diet_contrast.sort_values("delta_volatile_minus_stable", ascending=False).head(6)
            bot = diet_contrast.sort_values("delta_volatile_minus_stable", ascending=True).head(6)
            lines.append("### Diet (coarse)")
            for _, row in top.iterrows():
                lines.append(f"- ↑ {row['diet_coarse']}: Δ={row['delta_volatile_minus_stable']:.3f}")
            for _, row in bot.iterrows():
                lines.append(f"- ↓ {row['diet_coarse']}: Δ={row['delta_volatile_minus_stable']:.3f}")
        if len(mot_contrast) > 0:
            top = mot_contrast.sort_values("delta_volatile_minus_stable", ascending=False).head(6)
            bot = mot_contrast.sort_values("delta_volatile_minus_stable", ascending=True).head(6)
            lines.append("")
            lines.append("### Motility (coarse)")
            for _, row in top.iterrows():
                lines.append(f"- ↑ {row['motility_coarse']}: Δ={row['delta_volatile_minus_stable']:.3f}")
            for _, row in bot.iterrows():
                lines.append(f"- ↓ {row['motility_coarse']}: Δ={row['delta_volatile_minus_stable']:.3f}")
        if len(hab_contrast) > 0:
            top = hab_contrast.sort_values("delta_volatile_minus_stable", ascending=False).head(6)
            bot = hab_contrast.sort_values("delta_volatile_minus_stable", ascending=True).head(6)
            lines.append("")
            lines.append("### Life habit (coarse)")
            for _, row in top.iterrows():
                lines.append(f"- ↑ {row['habit_coarse']}: Δ={row['delta_volatile_minus_stable']:.3f}")
            for _, row in bot.iterrows():
                lines.append(f"- ↓ {row['habit_coarse']}: Δ={row['delta_volatile_minus_stable']:.3f}")

        if len(diet_pc) > 0 or len(mot_pc) > 0 or len(hab_pc) > 0:
            lines.extend(
                [
                    "",
                    "## Category-by-category (controls time)",
                    "",
                    "For each category, we test whether locality occupancy tracks volatility even after controlling for the strong",
                    "long-term time trend (partial correlation; permutation p-values). Only categories present in ≥12 bins are tested.",
                    "",
                ]
            )
        if len(diet_pc) > 0:
            lines.append("### Diet (coarse): partial corr(volatility, occupancy | time)")
            top = diet_pc.head(6)
            bot = diet_pc.sort_values("partial_corr", ascending=True).head(6)
            for _, row in top.iterrows():
                lines.append(
                    f"- ↑ {row['diet_coarse']}: r={row['partial_corr']:.3f}, p={row['partial_p_perm']:.3g}, n={int(row['n_bins'])}"
                )
            for _, row in bot.iterrows():
                lines.append(
                    f"- ↓ {row['diet_coarse']}: r={row['partial_corr']:.3f}, p={row['partial_p_perm']:.3g}, n={int(row['n_bins'])}"
                )
        if len(mot_pc) > 0:
            lines.append("")
            lines.append("### Motility (coarse): partial corr(volatility, occupancy | time)")
            top = mot_pc.head(6)
            bot = mot_pc.sort_values("partial_corr", ascending=True).head(6)
            for _, row in top.iterrows():
                lines.append(
                    f"- ↑ {row['motility_coarse']}: r={row['partial_corr']:.3f}, p={row['partial_p_perm']:.3g}, n={int(row['n_bins'])}"
                )
            for _, row in bot.iterrows():
                lines.append(
                    f"- ↓ {row['motility_coarse']}: r={row['partial_corr']:.3f}, p={row['partial_p_perm']:.3g}, n={int(row['n_bins'])}"
                )
        if len(hab_pc) > 0:
            lines.append("")
            lines.append("### Life habit (coarse): partial corr(volatility, occupancy | time)")
            top = hab_pc.head(6)
            bot = hab_pc.sort_values("partial_corr", ascending=True).head(6)
            for _, row in top.iterrows():
                lines.append(
                    f"- ↑ {row['habit_coarse']}: r={row['partial_corr']:.3f}, p={row['partial_p_perm']:.3g}, n={int(row['n_bins'])}"
                )
            for _, row in bot.iterrows():
                lines.append(
                    f"- ↓ {row['habit_coarse']}: r={row['partial_corr']:.3f}, p={row['partial_p_perm']:.3g}, n={int(row['n_bins'])}"
                )

        lines.extend(
            [
                "",
                "## Files",
                "",
                f"- Pairwise similarities: `{out_dir / 'pairwise_decomposition.csv'}`",
                f"- Time-bin metrics: `{out_dir / 'timebin_metrics_decomposition.csv'}`",
                f"- Category contrasts: `{out_dir}` (`*_contrast.csv`)",
                f"- Figures: `{fig_dir}`",
                "",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
