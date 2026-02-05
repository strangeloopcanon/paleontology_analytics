from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CoordsMode = Literal["paleo", "modern"]
Weighting = Literal["occurrence", "locality"]


MISSING_STRINGS = {"", "nan", "none", "null"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_missing_strings(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.mask(text.isin(MISSING_STRINGS), pd.NA)


def _analysis_coords(df: pd.DataFrame, *, mode: CoordsMode) -> pd.DataFrame:
    if mode == "modern":
        df["analysis_lat"] = df["lat"]
        df["analysis_lng"] = df["lng"]
        return df

    # paleo: prefer PBDB paleocoordinates, fallback to modern.
    df["analysis_lat"] = df["paleolat"].where(df["paleolat"].notna(), df["lat"])
    df["analysis_lng"] = df["paleolng"].where(df["paleolng"].notna(), df["lng"])
    return df


def _bin_localities(df: pd.DataFrame, *, bin_deg: float) -> pd.DataFrame:
    df["lat_bin"] = (df["analysis_lat"] / bin_deg).round() * bin_deg
    df["lng_bin"] = (df["analysis_lng"] / bin_deg).round() * bin_deg
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))
    return df


def _circular_mean_deg(longitudes_deg: np.ndarray) -> float:
    rad = np.deg2rad(longitudes_deg)
    s = np.nanmean(np.sin(rad))
    c = np.nanmean(np.cos(rad))
    return float(np.rad2deg(np.arctan2(s, c)))


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0088
    lat1 = np.deg2rad(lat1)
    lng1 = np.deg2rad(lng1)
    lat2 = np.deg2rad(lat2)
    lng2 = np.deg2rad(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def _unit_xyz(lat_deg: float, lng_deg: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lng = np.deg2rad(lng_deg)
    x = np.cos(lat) * np.cos(lng)
    y = np.cos(lat) * np.sin(lng)
    z = np.sin(lat)
    return np.array([x, y, z], dtype=float)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    c = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return float(np.arccos(c))


@dataclass(frozen=True)
class Scenario:
    name: str
    coords_mode: CoordsMode
    weighting: Weighting
    time_bin_myr: float
    locality_bin_deg: float
    max_delta_myr: float


def load_occurrences(
    data_path: str,
    *,
    coords_mode: CoordsMode,
    time_bin_myr: float,
    locality_bin_deg: float,
) -> pd.DataFrame:
    cols = [
        "source_db",
        "occurrence_id",
        "genus",
        "phylum",
        "environment",
        "mid_ma",
        "lat",
        "lng",
        "paleolat",
        "paleolng",
    ]
    df = pd.read_parquet(data_path, columns=cols)

    df["genus"] = _normalize_missing_strings(df["genus"])
    df["phylum"] = _normalize_missing_strings(df["phylum"])
    df["environment"] = _normalize_missing_strings(df["environment"])
    df = df.dropna(subset=["genus", "mid_ma"])

    # Deduplicate overlapping API pulls.
    df = df.drop_duplicates(subset=["source_db", "occurrence_id"]).copy()

    df["mid_ma"] = pd.to_numeric(df["mid_ma"], errors="coerce")
    for c in ["lat", "lng", "paleolat", "paleolng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mid_ma"])

    df = _analysis_coords(df, mode=coords_mode)
    df = df.dropna(subset=["analysis_lat", "analysis_lng"])

    df["time_bin"] = (df["mid_ma"] / time_bin_myr).round() * time_bin_myr
    df = _bin_localities(df, bin_deg=locality_bin_deg)
    return df


def compute_genus_bin_table(df: pd.DataFrame, *, weighting: Weighting) -> pd.DataFrame:
    # Two centroid definitions:
    # - occurrence: use all occurrences (median lat + circular mean lng)
    # - locality: deduplicate to unique localities per genus-bin before centroiding
    base = df
    if weighting == "locality":
        base = df.drop_duplicates(subset=["genus", "time_bin", "locality"]).copy()

    cent = (
        base.groupby(["genus", "time_bin"])
        .agg(
            centroid_lat=("analysis_lat", "median"),
            centroid_lng=("analysis_lng", lambda s: _circular_mean_deg(s.to_numpy(dtype=float))),
        )
        .reset_index()
    )

    # Features computed from the full (non-deduped) data in the bin.
    feats = (
        df.groupby(["genus", "time_bin"])
        .agg(
            abundance=("genus", "size"),
            geographic_range=("locality", "nunique"),
            lat_min=("analysis_lat", "min"),
            lat_max=("analysis_lat", "max"),
            env_breadth=("environment", "nunique"),
        )
        .reset_index()
    )
    feats["lat_range"] = feats["lat_max"] - feats["lat_min"]

    phylum_mode = (
        df.dropna(subset=["phylum"])
        .groupby("genus")["phylum"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
    )

    out = cent.merge(feats, on=["genus", "time_bin"], how="left")
    out["phylum"] = out["genus"].map(phylum_mode)
    out = out.sort_values(["genus", "time_bin"], ascending=[True, False]).reset_index(drop=True)
    out["age_bins"] = out.groupby("genus").cumcount()
    return out


def add_velocity_features(genus_bin: pd.DataFrame, *, max_delta_myr: float) -> pd.DataFrame:
    gb = genus_bin.copy()
    gb["prev_time_bin"] = gb.groupby("genus")["time_bin"].shift(1)
    gb["prev_lat"] = gb.groupby("genus")["centroid_lat"].shift(1)
    gb["prev_lng"] = gb.groupby("genus")["centroid_lng"].shift(1)
    gb["delta_myr"] = gb["prev_time_bin"] - gb["time_bin"]

    valid = gb[["prev_lat", "prev_lng", "centroid_lat", "centroid_lng", "delta_myr"]].notna().all(axis=1)
    valid &= gb["delta_myr"] > 0
    valid &= gb["delta_myr"] <= float(max_delta_myr)

    dist = np.full(len(gb), np.nan, dtype=float)
    idx = valid.to_numpy().nonzero()[0]
    for i in idx:
        dist[i] = _haversine_km(gb.loc[i, "prev_lat"], gb.loc[i, "prev_lng"], gb.loc[i, "centroid_lat"], gb.loc[i, "centroid_lng"])

    gb["distance_km_back"] = dist
    gb["velocity_km_per_myr_back"] = gb["distance_km_back"] / gb["delta_myr"]
    return gb


def compute_global_centroids(df: pd.DataFrame, *, weighting: Weighting) -> pd.DataFrame:
    base = df
    if weighting == "locality":
        base = df.drop_duplicates(subset=["time_bin", "locality"]).copy()

    out = (
        base.groupby("time_bin")
        .agg(
            global_centroid_lat=("analysis_lat", "median"),
            global_centroid_lng=("analysis_lng", lambda s: _circular_mean_deg(s.to_numpy(dtype=float))),
            n_occ=("analysis_lat", "size"),
            n_localities=("locality", "nunique"),
        )
        .reset_index()
        .sort_values("time_bin", ascending=False)
        .reset_index(drop=True)
    )

    out["prev_time_bin"] = out["time_bin"].shift(1)
    out["prev_lat"] = out["global_centroid_lat"].shift(1)
    out["prev_lng"] = out["global_centroid_lng"].shift(1)
    out["delta_myr"] = out["prev_time_bin"] - out["time_bin"]

    dist = []
    for _, row in out.iterrows():
        if pd.isna(row["prev_lat"]) or pd.isna(row["delta_myr"]) or row["delta_myr"] <= 0:
            dist.append(np.nan)
        else:
            dist.append(_haversine_km(row["prev_lat"], row["prev_lng"], row["global_centroid_lat"], row["global_centroid_lng"]))
    out["global_distance_km_back"] = dist
    out["global_velocity_km_per_myr_back"] = out["global_distance_km_back"] / out["delta_myr"]
    return out


def add_alignment_with_global(genus_bin: pd.DataFrame, global_bin: pd.DataFrame) -> pd.DataFrame:
    gb = genus_bin.copy()
    g = global_bin.set_index("time_bin")

    align = np.full(len(gb), np.nan, dtype=float)
    for i, row in gb.iterrows():
        if pd.isna(row["prev_lat"]) or pd.isna(row["prev_time_bin"]):
            continue
        t = float(row["time_bin"])
        if t not in g.index:
            continue
        g_row = g.loc[t]
        if pd.isna(g_row["prev_lat"]):
            continue

        # Approximate displacement direction in 3D as difference of unit vectors.
        u_prev = _unit_xyz(float(row["prev_lat"]), float(row["prev_lng"]))
        u_curr = _unit_xyz(float(row["centroid_lat"]), float(row["centroid_lng"]))
        d_tax = u_curr - u_prev

        g_prev = _unit_xyz(float(g_row["prev_lat"]), float(g_row["prev_lng"]))
        g_curr = _unit_xyz(float(g_row["global_centroid_lat"]), float(g_row["global_centroid_lng"]))
        d_glob = g_curr - g_prev

        # Cosine similarity proxy via angle.
        angle = _angle_between(d_tax, d_glob)
        align[i] = float(np.cos(angle))

    gb["align_with_global_sampling"] = align

    # Merge global velocity covariate.
    gb = gb.merge(
        global_bin[["time_bin", "global_velocity_km_per_myr_back", "n_occ", "n_localities"]],
        on="time_bin",
        how="left",
    )
    return gb


def build_discrete_time_hazard_dataset(genus_bin: pd.DataFrame) -> pd.DataFrame:
    # Determine the next younger bin globally.
    time_bins = sorted(genus_bin["time_bin"].unique(), reverse=True)
    next_bin = {time_bins[i]: time_bins[i + 1] for i in range(len(time_bins) - 1)}

    df = genus_bin.copy()
    df["next_time_bin"] = df["time_bin"].map(next_bin)
    df = df.dropna(subset=["next_time_bin"]).copy()  # exclude youngest bin (censored interval end)

    presence = genus_bin[["genus", "time_bin"]].drop_duplicates()
    presence = presence.rename(columns={"time_bin": "next_time_bin"}).assign(in_next=1)

    df = df.merge(presence, on=["genus", "next_time_bin"], how="left")
    df["event_extinct_next_bin"] = df["in_next"].isna().astype(int)
    df = df.drop(columns=["in_next"])
    return df


def fit_hazard_model(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    group_col: str,
    time_col: str,
    repeats: int,
    random_state: int,
) -> dict:
    X_num = data[feature_cols].astype(float)
    y = data["event_extinct_next_bin"].astype(int)
    groups = data[group_col]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
            ("time", OneHotEncoder(handle_unknown="ignore"), [time_col]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    rows = []
    for i in range(int(repeats)):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state + i)
        train_idx, test_idx = next(splitter.split(X_num, y, groups=groups))

        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        pipe.fit(train_df[feature_cols + [time_col]], train_df["event_extinct_next_bin"])
        prob = pipe.predict_proba(test_df[feature_cols + [time_col]])[:, 1]
        auc = float(roc_auc_score(test_df["event_extinct_next_bin"], prob))
        ap = float(average_precision_score(test_df["event_extinct_next_bin"], prob))

        # Extract numeric coefficients only (standardized).
        # Coeff order: numeric first, then time dummies.
        coef = pipe.named_steps["clf"].coef_.reshape(-1)
        num_coef = coef[: len(feature_cols)]

        row = {"repeat": i, "auc": auc, "avg_precision": ap}
        for name, c in zip(feature_cols, num_coef, strict=True):
            row[f"coef_{name}"] = float(c)
            row[f"or_{name}"] = float(np.exp(c))
        rows.append(row)

    rep = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(data)),
        "event_rate": float(data["event_extinct_next_bin"].mean()),
        "auc_mean": float(rep["auc"].mean()),
        "auc_sd": float(rep["auc"].std(ddof=1)) if len(rep) > 1 else 0.0,
        "auc_p2_5": float(rep["auc"].quantile(0.025)),
        "auc_p97_5": float(rep["auc"].quantile(0.975)),
        "ap_mean": float(rep["avg_precision"].mean()),
        "ap_sd": float(rep["avg_precision"].std(ddof=1)) if len(rep) > 1 else 0.0,
    }

    coef_rows = []
    for name in feature_cols:
        coef_rows.append(
            {
                "feature": name,
                "coef_mean": float(rep[f"coef_{name}"].mean()),
                "coef_p2_5": float(rep[f"coef_{name}"].quantile(0.025)),
                "coef_p97_5": float(rep[f"coef_{name}"].quantile(0.975)),
                "odds_ratio_mean": float(rep[f"or_{name}"].mean()),
                "odds_ratio_p2_5": float(rep[f"or_{name}"].quantile(0.025)),
                "odds_ratio_p97_5": float(rep[f"or_{name}"].quantile(0.975)),
            }
        )
    coef_summary = pd.DataFrame(coef_rows).sort_values("odds_ratio_mean", ascending=False)

    return {"repeats": rep, "summary": summary, "coef_summary": coef_summary}


def _cap_series(s: pd.Series, q: float) -> tuple[pd.Series, float]:
    cap = float(s.quantile(q))
    return s.clip(upper=cap), cap


def run_scenario(
    scenario: Scenario,
    *,
    data_path: str,
    out_dir: Path,
    repeats: int,
    random_state: int,
    outlier_q: float,
) -> dict:
    df = load_occurrences(
        data_path,
        coords_mode=scenario.coords_mode,
        time_bin_myr=scenario.time_bin_myr,
        locality_bin_deg=scenario.locality_bin_deg,
    )
    genus_bin = compute_genus_bin_table(df, weighting=scenario.weighting)
    genus_bin = add_velocity_features(genus_bin, max_delta_myr=scenario.max_delta_myr)

    global_bin = compute_global_centroids(df, weighting=scenario.weighting)
    genus_bin = add_alignment_with_global(genus_bin, global_bin)

    # Capping velocity outliers (winsorization).
    genus_bin["velocity_km_per_myr_back"], cap = _cap_series(genus_bin["velocity_km_per_myr_back"], outlier_q)

    hazard = build_discrete_time_hazard_dataset(genus_bin)

    # Features for the hazard model.
    features = [
        "velocity_km_per_myr_back",
        "abundance",
        "geographic_range",
        "lat_range",
        "env_breadth",
        "age_bins",
        "global_velocity_km_per_myr_back",
        "n_occ",
        "n_localities",
        "align_with_global_sampling",
    ]

    # Drop rows missing any required features.
    hazard = hazard.dropna(subset=features).copy()

    # Baseline (no mobility feature).
    baseline_features = [c for c in features if c != "velocity_km_per_myr_back"]

    fit_full = fit_hazard_model(
        hazard,
        feature_cols=features,
        group_col="genus",
        time_col="time_bin",
        repeats=repeats,
        random_state=random_state,
    )
    fit_base = fit_hazard_model(
        hazard,
        feature_cols=baseline_features,
        group_col="genus",
        time_col="time_bin",
        repeats=repeats,
        random_state=random_state,
    )

    # Save outputs.
    scenario_dir = out_dir / scenario.name
    _ensure_dir(scenario_dir)
    _ensure_dir(scenario_dir / "tables")
    _ensure_dir(scenario_dir / "figures")
    _ensure_dir(scenario_dir / "results")

    genus_bin.to_parquet(scenario_dir / "results" / "genus_bin.parquet", index=False)
    global_bin.to_csv(scenario_dir / "results" / "global_sampling_centroids.csv", index=False)
    hazard.to_parquet(scenario_dir / "results" / "hazard_dataset.parquet", index=False)

    fit_full["repeats"].to_csv(scenario_dir / "results" / "hazard_model_repeats.csv", index=False)
    fit_full["coef_summary"].to_csv(scenario_dir / "results" / "hazard_model_coef_summary.csv", index=False)
    fit_base["coef_summary"].to_csv(scenario_dir / "results" / "hazard_model_baseline_coef_summary.csv", index=False)

    meta = {
        "scenario": scenario.__dict__,
        "n_occ": int(len(df)),
        "n_genus_bin_rows": int(len(genus_bin)),
        "n_hazard_rows": int(len(hazard)),
        "velocity_cap_q": outlier_q,
        "velocity_cap_value": cap,
        "full_model": fit_full["summary"],
        "baseline_model": fit_base["summary"],
        "delta_auc_full_minus_baseline": float(fit_full["summary"]["auc_mean"] - fit_base["summary"]["auc_mean"]),
    }
    (scenario_dir / "results" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Figure: coefficient ORs for core predictors.
    coef = fit_full["coef_summary"].set_index("feature")
    core = [
        "velocity_km_per_myr_back",
        "abundance",
        "geographic_range",
        "lat_range",
        "env_breadth",
        "age_bins",
    ]
    plot = coef.loc[[c for c in core if c in coef.index]].copy()
    plot = plot.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        plot["odds_ratio_mean"],
        np.arange(len(plot)),
        xerr=[plot["odds_ratio_mean"] - plot["odds_ratio_p2_5"], plot["odds_ratio_p97_5"] - plot["odds_ratio_mean"]],
        fmt="o",
        color="#1f77b4",
        ecolor="gray",
        capsize=3,
    )
    ax.axvline(1.0, color="black", linewidth=1, alpha=0.6)
    ax.set_yticks(np.arange(len(plot)))
    ax.set_yticklabels(plot.index.tolist())
    ax.set_xlabel("Odds ratio per 1 SD (mean ± 95% across splits)")
    ax.set_title(f"Hazard model effects ({scenario.name})")
    fig.tight_layout()
    fig.savefig(scenario_dir / "figures" / "coef_odds_ratios.png", dpi=200)
    plt.close(fig)

    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--out", default="thesis/paleobiotic_velocity/output")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outlier-q", type=float, default=0.995)
    args = p.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    scenarios = [
        Scenario(
            name="paleo_occurrence_5myr_5deg",
            coords_mode="paleo",
            weighting="occurrence",
            time_bin_myr=5.0,
            locality_bin_deg=5.0,
            max_delta_myr=10.0,
        ),
        Scenario(
            name="paleo_locality_5myr_5deg",
            coords_mode="paleo",
            weighting="locality",
            time_bin_myr=5.0,
            locality_bin_deg=5.0,
            max_delta_myr=10.0,
        ),
        Scenario(
            name="modern_occurrence_5myr_5deg_negative_control",
            coords_mode="modern",
            weighting="occurrence",
            time_bin_myr=5.0,
            locality_bin_deg=5.0,
            max_delta_myr=10.0,
        ),
        Scenario(
            name="paleo_locality_5myr_10deg_sensitivity",
            coords_mode="paleo",
            weighting="locality",
            time_bin_myr=5.0,
            locality_bin_deg=10.0,
            max_delta_myr=10.0,
        ),
    ]

    all_meta = []
    for s in scenarios:
        print(f"Running scenario: {s.name}")
        meta = run_scenario(
            s,
            data_path=args.data,
            out_dir=out_dir,
            repeats=args.repeats,
            random_state=args.seed,
            outlier_q=float(args.outlier_q),
        )
        all_meta.append(meta)

    (out_dir / "summary.json").write_text(json.dumps(all_meta, indent=2) + "\n")

    summary_rows = []
    for m in all_meta:
        row = {
            "scenario": m["scenario"]["name"],
            "coords_mode": m["scenario"]["coords_mode"],
            "weighting": m["scenario"]["weighting"],
            "time_bin_myr": m["scenario"]["time_bin_myr"],
            "locality_bin_deg": m["scenario"]["locality_bin_deg"],
            "n_occ": m["n_occ"],
            "n_genus_bin_rows": m["n_genus_bin_rows"],
            "n_hazard_rows": m["n_hazard_rows"],
            "auc_full_mean": m["full_model"]["auc_mean"],
            "auc_base_mean": m["baseline_model"]["auc_mean"],
            "delta_auc": m["delta_auc_full_minus_baseline"],
            "velocity_cap_value": m["velocity_cap_value"],
        }
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    print(f"Wrote: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
