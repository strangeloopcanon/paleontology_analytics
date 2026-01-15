from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    # CSV-escaped quotes from PBDB often look like ""photoautotroph"" inside a longer string.
    s = s.strip('"').strip("'").strip()
    s = s.replace('"', "").strip()
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


def _coarse_diet(x: Any) -> str | None:
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
    if "photosymbiotic" in t or "photoautotroph" in t:
        return "photoautotroph"
    return s


def _coarse_motility(x: Any) -> str | None:
    s = _clean_name(x)
    if s is None:
        return None
    t = s.lower()
    if "actively mobile" in t:
        return "actively mobile"
    if "facultatively mobile" in t:
        return "facultatively mobile"
    if "fast-moving" in t:
        return "fast-moving"
    if "slow-moving" in t:
        return "slow-moving"
    if "stationary" in t:
        return "stationary"
    if "passively mobile" in t:
        return "passively mobile"
    return s


def _coarse_habit(x: Any) -> str | None:
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
    if "boring" in t:
        return "boring"
    return s


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yy = y[mask]
    XX = X[mask]
    if len(yy) < 4:
        out = np.full_like(y, fill_value=np.nan, dtype=float)
        out[mask] = np.nan
        return out
    A = np.column_stack([np.ones(len(yy)), XX])
    coef, _, _, _ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A.dot(coef)
    out = np.full_like(y, fill_value=np.nan, dtype=float)
    out[mask] = resid
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 4:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


@dataclass(frozen=True)
class ShiftCorrResult:
    corr: float
    p_shift_exact: float
    n_bins: int


def _partial_corr_shift_exact(
    *,
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
) -> ShiftCorrResult:
    x = x.astype(float)
    y = y.astype(float)
    controls = controls.astype(float)
    n = int(len(x))
    if n < 8 or len(y) != n or controls.shape[0] != n:
        return ShiftCorrResult(corr=float("nan"), p_shift_exact=float("nan"), n_bins=n)

    y_res = _ols_residuals(y, controls)
    x_res = _ols_residuals(x, controls)
    obs = _corr(x_res, y_res)
    if not np.isfinite(obs):
        return ShiftCorrResult(corr=float("nan"), p_shift_exact=float("nan"), n_bins=n)

    corrs: list[float] = []
    for shift in range(n):
        xs = np.roll(x, shift)
        xs_res = _ols_residuals(xs, controls)
        corrs.append(_corr(xs_res, y_res))

    arr = np.asarray(corrs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) != n:
        return ShiftCorrResult(corr=obs, p_shift_exact=float("nan"), n_bins=n)
    p = float(np.mean(np.abs(arr) >= abs(obs)))
    return ShiftCorrResult(corr=obs, p_shift_exact=p, n_bins=n)


def _compute_occupancy(
    df: pd.DataFrame,
    *,
    cat_col: str,
    totals: pd.DataFrame,
) -> pd.DataFrame:
    d = df[["time_bin", "locality", cat_col]].dropna().copy()
    d = d.drop_duplicates(subset=["time_bin", "locality", cat_col]).copy()
    occ = d.groupby(["time_bin", cat_col])["locality"].nunique().rename("n_localities_with_cat").reset_index()
    out = occ.merge(totals, on="time_bin", how="left")
    out["occupancy_frac"] = out["n_localities_with_cat"] / out["n_localities"]
    return out


def _compute_mean_locality_frac(
    df: pd.DataFrame,
    *,
    cat_col: str,
    locality_totals: pd.DataFrame,
) -> pd.DataFrame:
    d = df[["time_bin", "locality", "genus", cat_col]].dropna().copy()
    counts = (
        d.groupby(["time_bin", "locality", cat_col])["genus"].nunique().rename("n_genera_cat").reset_index()
    )
    counts = counts.merge(locality_totals, on=["time_bin", "locality"], how="left")
    counts["mean_locality_frac"] = counts["n_genera_cat"] / counts["n_genera"]
    out = (
        counts.groupby(["time_bin", cat_col])["mean_locality_frac"].mean().rename("mean_locality_frac").reset_index()
    )
    return out


def _wide_from_long(
    occ: pd.DataFrame, *, cat_col: str, bins: list[float], fill: float = 0.0
) -> tuple[pd.DataFrame, list[str]]:
    value_col = "occupancy_frac" if "occupancy_frac" in occ.columns else "mean_locality_frac"
    wide = occ.pivot_table(index="time_bin", columns=cat_col, values=value_col, fill_value=fill, aggfunc="mean")
    wide = wide.reindex(bins).fillna(fill)
    cats = [str(c) for c in wide.columns]
    wide.columns = cats
    wide = wide.reset_index()
    return wide, cats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="thesis/synthesis/output_role_jobs_volatility_v1")
    p.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--mapping", default="thesis/convergence/output_v3_fullpbdb/ecospace_genus_mapping.csv")
    p.add_argument("--pair-level-merged", default="thesis/synthesis/output_pair_level_model_volatility_v1/merged_pairs.csv")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--grid-deg", type=float, default=10.0)
    p.add_argument("--min-genera-per-region", type=int, default=25)
    p.add_argument("--min-bins-per-category", type=int, default=12)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # Bin-level controls aligned to the publication-oriented pair-level model.
    pair_cols = [
        "time_bin",
        "vol_z",
        "time_z",
        "prov_z",
        "pc1_z",
        "pc2_z",
        "delta_from_prev_T_field_meanabs",
        "provinciality",
    ]
    pairs = pd.read_csv(args.pair_level_merged, usecols=pair_cols)
    bins_df = pairs.drop_duplicates(subset=["time_bin"]).sort_values("time_bin", ascending=False).reset_index(drop=True)
    bins_df["time_bin"] = pd.to_numeric(bins_df["time_bin"], errors="coerce")
    bins_df = bins_df.dropna(subset=["time_bin"]).copy()
    bins = [float(x) for x in bins_df["time_bin"].tolist()]
    if len(bins) < 8:
        raise SystemExit("Too few bins found in pair-level merged file; cannot proceed.")
    bins_df.to_csv(out_dir / "bin_controls.csv", index=False)

    vol_raw = bins_df["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    vol_z = bins_df["vol_z"].to_numpy(dtype=float)
    controls = bins_df[["time_z", "pc1_z", "pc2_z", "prov_z"]].to_numpy(dtype=float)

    q25 = float(np.nanquantile(vol_raw, 0.25))
    q75 = float(np.nanquantile(vol_raw, 0.75))
    stable_bins = set(bins_df.loc[bins_df["delta_from_prev_T_field_meanabs"] <= q25, "time_bin"].tolist())
    volatile_bins = set(bins_df.loc[bins_df["delta_from_prev_T_field_meanabs"] >= q75, "time_bin"].tolist())

    # Load PBDB occurrences and ecospace mapping.
    occ_cols = [
        "source_db",
        "occurrence_id",
        "mid_ma",
        "lat",
        "lng",
        "paleolat",
        "paleolng",
        "genus",
    ]
    occ = pd.read_parquet(args.pbdb, columns=occ_cols)
    occ = occ[occ["source_db"] == "PBDB"].drop_duplicates(subset=["source_db", "occurrence_id"]).copy()
    occ["genus"] = occ["genus"].map(_clean_name)
    occ = occ.dropna(subset=["genus", "mid_ma"]).copy()

    mapping = pd.read_csv(args.mapping)
    for c in ["genus", "jev", "jdt", "jmo", "jlh", "role_id"]:
        mapping[c] = mapping[c].map(_clean_name)
    mapping = mapping.dropna(subset=["genus", "role_id"]).copy()
    mapping["diet_coarse"] = mapping["jdt"].map(_coarse_diet)
    mapping["motility_coarse"] = mapping["jmo"].map(_coarse_motility)
    mapping["habit_coarse"] = mapping["jlh"].map(_coarse_habit)

    df = occ.merge(mapping, on="genus", how="left")
    df = df[df["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()
    df = df.dropna(subset=["role_id", "diet_coarse", "motility_coarse", "habit_coarse"]).copy()
    df = _analysis_lat_lng(df)
    df["time_bin"] = (pd.to_numeric(df["mid_ma"], errors="coerce") / float(args.time_bin_myr)).round() * float(
        args.time_bin_myr
    )
    df = df[df["time_bin"].isin(bins)].copy()
    df["lat_bin"] = (df["analysis_lat"] / float(args.grid_deg)).round() * float(args.grid_deg)
    df["lng_bin"] = (df["analysis_lng"] / float(args.grid_deg)).round() * float(args.grid_deg)
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))
    df = df.drop_duplicates(subset=["time_bin", "locality", "genus"]).copy()

    loc_counts = df.groupby(["time_bin", "locality"])["genus"].nunique().rename("n_genera").reset_index()
    allowed = loc_counts[loc_counts["n_genera"] >= int(args.min_genera_per_region)].copy()
    totals = allowed.groupby("time_bin")["locality"].nunique().rename("n_localities").reset_index()

    df = df.merge(allowed[["time_bin", "locality"]], on=["time_bin", "locality"], how="inner")

    totals = totals[totals["time_bin"].isin(bins)].copy()
    totals.to_csv(out_dir / "bin_localities.csv", index=False)
    allowed.to_csv(out_dir / "bin_locality_genera.csv", index=False)

    # Occupancy tables (long).
    occ_diet = _compute_occupancy(df, cat_col="diet_coarse", totals=totals)
    occ_mot = _compute_occupancy(df, cat_col="motility_coarse", totals=totals)
    occ_hab = _compute_occupancy(df, cat_col="habit_coarse", totals=totals)
    occ_role = _compute_occupancy(df, cat_col="role_id", totals=totals)

    occ_diet.to_csv(out_dir / "diet_occupancy_long.csv", index=False)
    occ_mot.to_csv(out_dir / "motility_occupancy_long.csv", index=False)
    occ_hab.to_csv(out_dir / "habit_occupancy_long.csv", index=False)
    occ_role.to_csv(out_dir / "role_occupancy_long.csv", index=False)

    # Composition tables (long): mean share of genera-per-locality in each category.
    frac_diet = _compute_mean_locality_frac(df, cat_col="diet_coarse", locality_totals=allowed)
    frac_mot = _compute_mean_locality_frac(df, cat_col="motility_coarse", locality_totals=allowed)
    frac_hab = _compute_mean_locality_frac(df, cat_col="habit_coarse", locality_totals=allowed)
    frac_role = _compute_mean_locality_frac(df, cat_col="role_id", locality_totals=allowed)

    frac_diet.to_csv(out_dir / "diet_mean_locality_frac_long.csv", index=False)
    frac_mot.to_csv(out_dir / "motility_mean_locality_frac_long.csv", index=False)
    frac_hab.to_csv(out_dir / "habit_mean_locality_frac_long.csv", index=False)
    frac_role.to_csv(out_dir / "role_mean_locality_frac_long.csv", index=False)

    def _run_assoc(occ_long: pd.DataFrame, *, cat_col: str) -> pd.DataFrame:
        wide, cats = _wide_from_long(occ_long, cat_col=cat_col, bins=bins, fill=0.0)
        rows: list[dict[str, Any]] = []
        for cat in cats:
            y = wide[cat].to_numpy(dtype=float)
            present_bins = int(np.sum(y > 0))
            if present_bins < int(args.min_bins_per_category):
                continue
            res = _partial_corr_shift_exact(x=vol_z, y=y, controls=controls)
            y_stable = wide.loc[wide["time_bin"].isin(stable_bins), cat].to_numpy(dtype=float)
            y_volatile = wide.loc[wide["time_bin"].isin(volatile_bins), cat].to_numpy(dtype=float)
            delta = float(np.nanmean(y_volatile) - np.nanmean(y_stable)) if len(y_stable) and len(y_volatile) else float(
                "nan"
            )
            rows.append(
                {
                    cat_col: cat,
                    "present_bins": present_bins,
                    "partial_corr": float(res.corr),
                    "p_shift_exact": float(res.p_shift_exact),
                    "delta_occ_q75_minus_q25": delta,
                }
            )
        out = pd.DataFrame(rows)
        if len(out) == 0:
            return out
        out = out.sort_values(["p_shift_exact", "partial_corr"], ascending=[True, False]).reset_index(drop=True)

        # Multiple-testing helper (Benjamini–Hochberg FDR on shift p-values).
        pv = out["p_shift_exact"].to_numpy(dtype=float)
        n = len(pv)
        order = np.argsort(pv)
        ranks = np.arange(1, n + 1, dtype=float)
        q = np.empty(n, dtype=float)
        q_sorted = pv[order] * float(n) / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q[order] = np.clip(q_sorted, 0.0, 1.0)
        out["q_bh_fdr"] = q
        return out

    assoc_occ_diet = _run_assoc(occ_diet, cat_col="diet_coarse")
    assoc_occ_mot = _run_assoc(occ_mot, cat_col="motility_coarse")
    assoc_occ_hab = _run_assoc(occ_hab, cat_col="habit_coarse")
    assoc_occ_role = _run_assoc(occ_role, cat_col="role_id")

    assoc_frac_diet = _run_assoc(frac_diet, cat_col="diet_coarse")
    assoc_frac_mot = _run_assoc(frac_mot, cat_col="motility_coarse")
    assoc_frac_hab = _run_assoc(frac_hab, cat_col="habit_coarse")
    assoc_frac_role = _run_assoc(frac_role, cat_col="role_id")

    assoc_occ_diet.to_csv(out_dir / "diet_occupancy_volatility_assoc.csv", index=False)
    assoc_occ_mot.to_csv(out_dir / "motility_occupancy_volatility_assoc.csv", index=False)
    assoc_occ_hab.to_csv(out_dir / "habit_occupancy_volatility_assoc.csv", index=False)
    assoc_occ_role.to_csv(out_dir / "role_occupancy_volatility_assoc.csv", index=False)

    assoc_frac_diet.to_csv(out_dir / "diet_meanfrac_volatility_assoc.csv", index=False)
    assoc_frac_mot.to_csv(out_dir / "motility_meanfrac_volatility_assoc.csv", index=False)
    assoc_frac_hab.to_csv(out_dir / "habit_meanfrac_volatility_assoc.csv", index=False)
    assoc_frac_role.to_csv(out_dir / "role_meanfrac_volatility_assoc.csv", index=False)

    # Minimal markdown summary (full details in CSVs).
    def _top_lines(df_assoc: pd.DataFrame, label_col: str, k: int, *, positive: bool) -> list[str]:
        if len(df_assoc) == 0:
            return ["(none)"]
        lines = []
        dd = df_assoc[df_assoc["partial_corr"] > 0] if positive else df_assoc[df_assoc["partial_corr"] < 0]
        dd = dd.sort_values(["p_shift_exact", "partial_corr"], ascending=[True, False] if positive else [True, True])
        for _, r in dd.head(k).iterrows():
            lines.append(
                f"- {r[label_col]}: r={r['partial_corr']:.3f}, p_shift={r['p_shift_exact']:.3f}, q={r['q_bh_fdr']:.3f}, Δq75-q25={r['delta_occ_q75_minus_q25']:.3f}"
            )
        return lines

    summary_lines = [
        "# Role/job drivers under volatility (sampling+autocorr-aware)\n",
        "This asks: as climate volatility rises, which ecospace categories change in (i) geographic ubiquity and (ii) average within-locality composition?",
        "",
        f"- bins: {len(bins)} (from pair-level model)",
        f"- volatility quartiles (mean |ΔT| field): q25={q25:.3f}, q75={q75:.3f}",
        f"- controls: time_z + sampling_pc1_z + sampling_pc2_z + provinciality_z",
        "",
        "## Diet (coarse): geographic ubiquity (occupancy fraction)",
        "### Increases with volatility",
        *_top_lines(assoc_occ_diet, "diet_coarse", 6, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_occ_diet, "diet_coarse", 6, positive=False),
        "",
        "## Diet (coarse): mean within-locality share",
        "### Increases with volatility",
        *_top_lines(assoc_frac_diet, "diet_coarse", 6, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_frac_diet, "diet_coarse", 6, positive=False),
        "",
        "## Motility (coarse): geographic ubiquity",
        "### Increases with volatility",
        *_top_lines(assoc_occ_mot, "motility_coarse", 6, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_occ_mot, "motility_coarse", 6, positive=False),
        "",
        "## Motility (coarse): mean within-locality share",
        "### Increases with volatility",
        *_top_lines(assoc_frac_mot, "motility_coarse", 6, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_frac_mot, "motility_coarse", 6, positive=False),
        "",
        "## Life habit (coarse): geographic ubiquity",
        "### Increases with volatility",
        *_top_lines(assoc_occ_hab, "habit_coarse", 6, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_occ_hab, "habit_coarse", 6, positive=False),
        "",
        "## Life habit (coarse): mean within-locality share",
        "### Increases with volatility",
        *_top_lines(assoc_frac_hab, "habit_coarse", 6, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_frac_hab, "habit_coarse", 6, positive=False),
        "",
        "## Full roles: geographic ubiquity",
        "### Increases with volatility",
        *_top_lines(assoc_occ_role, "role_id", 8, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_occ_role, "role_id", 8, positive=False),
        "",
        "## Full roles: mean within-locality share",
        "### Increases with volatility",
        *_top_lines(assoc_frac_role, "role_id", 8, positive=True),
        "### Decreases with volatility",
        *_top_lines(assoc_frac_role, "role_id", 8, positive=False),
        "",
        "## Outputs",
        f"- bin controls: `{out_dir / 'bin_controls.csv'}`",
        f"- locality totals: `{out_dir / 'bin_localities.csv'}`, `{out_dir / 'bin_locality_genera.csv'}`",
        f"- long (occupancy): `{out_dir / 'diet_occupancy_long.csv'}`, `{out_dir / 'motility_occupancy_long.csv'}`, `{out_dir / 'habit_occupancy_long.csv'}`, `{out_dir / 'role_occupancy_long.csv'}`",
        f"- long (mean locality fractions): `{out_dir / 'diet_mean_locality_frac_long.csv'}`, `{out_dir / 'motility_mean_locality_frac_long.csv'}`, `{out_dir / 'habit_mean_locality_frac_long.csv'}`, `{out_dir / 'role_mean_locality_frac_long.csv'}`",
        f"- assoc (occupancy): `{out_dir / 'diet_occupancy_volatility_assoc.csv'}`, `{out_dir / 'motility_occupancy_volatility_assoc.csv'}`, `{out_dir / 'habit_occupancy_volatility_assoc.csv'}`, `{out_dir / 'role_occupancy_volatility_assoc.csv'}`",
        f"- assoc (mean locality fractions): `{out_dir / 'diet_meanfrac_volatility_assoc.csv'}`, `{out_dir / 'motility_meanfrac_volatility_assoc.csv'}`, `{out_dir / 'habit_meanfrac_volatility_assoc.csv'}`, `{out_dir / 'role_meanfrac_volatility_assoc.csv'}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")

    meta = {
        "n_bins": int(len(bins)),
        "controls": ["time_z", "pc1_z", "pc2_z", "prov_z"],
        "volatility_col_raw": "delta_from_prev_T_field_meanabs",
        "volatility_quartiles": {"q25": q25, "q75": q75},
        "min_bins_per_category": int(args.min_bins_per_category),
        "min_genera_per_region": int(args.min_genera_per_region),
        "grid_deg": float(args.grid_deg),
        "time_bin_myr": float(args.time_bin_myr),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
