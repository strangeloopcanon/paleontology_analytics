"""
Paleo-velocity analysis: infer genus mobility from reconstructed paleocoordinates.

Core idea:
- Aggregate occurrences into genus × time-bin paleogeographic centroids.
- Compute stepwise great-circle displacement between successive bins.
- Summarize mobility through time and test association with terminal (right-censored) extinction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.geo import add_analysis_coordinates, add_binned_locality


MISSING_STRINGS = {"", "nan", "none", "null"}


def _normalize_missing_strings(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.mask(text.isin(MISSING_STRINGS), pd.NA)


def _haversine_km(
    lat1_deg: np.ndarray,
    lng1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lng2_deg: np.ndarray,
) -> np.ndarray:
    """
    Vectorized great-circle distance (km) for arrays in degrees.
    """
    radius_km = 6371.0088
    lat1 = np.deg2rad(lat1_deg)
    lng1 = np.deg2rad(lng1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lng2 = np.deg2rad(lng2_deg)

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius_km * c


def _get_temperature_c(age_ma: float) -> float:
    """
    Approximate Phanerozoic global temperature (°C) from a small knot set.

    This matches the dashboard's simplified curve (Scotese/Veizer-style approximation),
    and is used only for descriptive overlays/correlation (not causal claims).
    """
    temp_points = {
        0: 14,
        10: 14,
        30: 18,
        50: 24,
        65: 22,
        80: 24,
        100: 26,
        140: 22,
        170: 20,
        200: 19,
        230: 22,
        250: 25,
        270: 16,
        300: 12,
        340: 14,
        360: 20,
        400: 22,
        420: 20,
        440: 16,
        450: 12,
        480: 18,
        500: 22,
        540: 24,
    }

    times = sorted(temp_points.keys())
    if age_ma <= times[0]:
        return float(temp_points[times[0]])
    if age_ma >= times[-1]:
        return float(temp_points[times[-1]])

    for t1, t2 in zip(times, times[1:]):
        if t1 <= age_ma <= t2:
            temp1, temp2 = temp_points[t1], temp_points[t2]
            frac = (age_ma - t1) / (t2 - t1)
            return float(temp1 + (temp2 - temp1) * frac)

    return 14.0


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PaleovelocityOutputs:
    output_dir: Path
    genus_bin_path: Path
    velocity_timeseries_path: Path
    top_movers_path: Path
    model_coefficients_path: Path
    model_repeats_path: Path
    model_coefficients_summary_path: Path
    model_metrics_path: Path
    fig_velocity_timeseries_path: Path
    fig_terminal_vs_nont_path: Path


def run_paleovelocity_study(
    *,
    data_path: str = "data/processed/merged_occurrences.parquet",
    output_dir: str = "thesis/archive/paleovelocity_pilot/output",
    time_bin_width_myr: float = 5.0,
    locality_bin_degrees: float = 5.0,
    max_delta_myr_for_velocity: float = 10.0,
    min_occurrences_per_genus_bin: int = 1,
    outlier_velocity_quantile: float = 0.995,
    n_model_repeats: int = 25,
    random_state: int = 42,
) -> PaleovelocityOutputs:
    """
    End-to-end analysis producing tables + figures suitable for a manuscript draft.
    """
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    tab_dir = out_dir / "tables"
    _ensure_dirs([out_dir, fig_dir, res_dir, tab_dir])

    outputs = PaleovelocityOutputs(
        output_dir=out_dir,
        genus_bin_path=res_dir / "genus_bin_features.parquet",
        velocity_timeseries_path=res_dir / "velocity_timeseries.csv",
        top_movers_path=tab_dir / "top_movers.csv",
        model_coefficients_path=res_dir / "terminal_extinction_logit_coefficients.csv",
        model_repeats_path=res_dir / "terminal_extinction_logit_repeats.csv",
        model_coefficients_summary_path=res_dir / "terminal_extinction_logit_coefficients_summary.csv",
        model_metrics_path=res_dir / "terminal_extinction_logit_metrics.json",
        fig_velocity_timeseries_path=fig_dir / "fig1_velocity_timeseries.png",
        fig_terminal_vs_nont_path=fig_dir / "fig2_terminal_vs_nont_velocity.png",
    )

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
    n_rows_loaded = int(len(df))

    df["genus"] = _normalize_missing_strings(df["genus"])
    df["phylum"] = _normalize_missing_strings(df["phylum"])
    df["environment"] = _normalize_missing_strings(df["environment"])
    df = df.dropna(subset=["genus", "mid_ma"])
    n_rows_after_clean = int(len(df))

    # PBDB downloads can overlap (multiple API pulls). Keep unique occurrence IDs per source DB.
    df = df.drop_duplicates(subset=["source_db", "occurrence_id"]).copy()
    n_rows_after_dedup = int(len(df))

    df["mid_ma"] = pd.to_numeric(df["mid_ma"], errors="coerce")
    for c in ["lat", "lng", "paleolat", "paleolng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mid_ma"])

    df = add_analysis_coordinates(df)
    df = df.dropna(subset=["analysis_lat", "analysis_lng"])
    n_rows_after_coords = int(len(df))
    df["time_bin"] = (df["mid_ma"] / time_bin_width_myr).round() * time_bin_width_myr

    df = add_binned_locality(
        df,
        lat_col="analysis_lat",
        lng_col="analysis_lng",
        bin_degrees=locality_bin_degrees,
        locality_col="locality",
    )

    # Circular mean for longitude (avoid dateline issues).
    lng_rad = np.deg2rad(df["analysis_lng"].to_numpy())
    df["sin_lng"] = np.sin(lng_rad)
    df["cos_lng"] = np.cos(lng_rad)

    grouped = (
        df.groupby(["genus", "time_bin"])
        .agg(
            abundance=("genus", "size"),
            geographic_range=("locality", "nunique"),
            lat_min=("analysis_lat", "min"),
            lat_max=("analysis_lat", "max"),
            centroid_lat=("analysis_lat", "median"),
            sin_mean=("sin_lng", "mean"),
            cos_mean=("cos_lng", "mean"),
            env_breadth=("environment", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["abundance"] >= int(min_occurrences_per_genus_bin)].copy()
    grouped["lat_range"] = grouped["lat_max"] - grouped["lat_min"]
    grouped["centroid_lng"] = np.rad2deg(np.arctan2(grouped["sin_mean"], grouped["cos_mean"]))

    # Per-genus phylum label (mode).
    phylum_mode = (
        df.dropna(subset=["phylum"])
        .groupby("genus")["phylum"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
    )
    grouped["phylum"] = grouped["genus"].map(phylum_mode)

    # Sort for stepwise deltas and age-in-bins.
    grouped = grouped.sort_values(["genus", "time_bin"], ascending=[True, False]).reset_index(drop=True)
    grouped["age_bins"] = grouped.groupby("genus").cumcount()

    # Stepwise displacement from previous (older) bin -> current bin.
    grouped["prev_time_bin"] = grouped.groupby("genus")["time_bin"].shift(1)
    grouped["prev_lat"] = grouped.groupby("genus")["centroid_lat"].shift(1)
    grouped["prev_lng"] = grouped.groupby("genus")["centroid_lng"].shift(1)
    grouped["delta_myr"] = grouped["prev_time_bin"] - grouped["time_bin"]

    valid = grouped[["prev_lat", "prev_lng", "centroid_lat", "centroid_lng", "delta_myr"]].notna().all(axis=1)
    valid &= grouped["delta_myr"] > 0
    valid &= grouped["delta_myr"] <= float(max_delta_myr_for_velocity)

    dist_km = np.full(len(grouped), np.nan, dtype=float)
    dist_km[valid.to_numpy()] = _haversine_km(
        grouped.loc[valid, "prev_lat"].to_numpy(dtype=float),
        grouped.loc[valid, "prev_lng"].to_numpy(dtype=float),
        grouped.loc[valid, "centroid_lat"].to_numpy(dtype=float),
        grouped.loc[valid, "centroid_lng"].to_numpy(dtype=float),
    )
    grouped["distance_km"] = dist_km
    grouped["velocity_km_per_myr"] = grouped["distance_km"] / grouped["delta_myr"]

    # Right-censoring: dataset is interval-limited; don't treat the youngest global bin as extinction.
    youngest_bin = float(grouped["time_bin"].min())
    grouped["terminal_bin"] = grouped["time_bin"] == grouped.groupby("genus")["time_bin"].transform("min")
    grouped["censored_at_youngest_bin"] = grouped.groupby("genus")["time_bin"].transform("min") == youngest_bin

    # Save the full genus-bin feature table for reproducibility.
    grouped.to_parquet(outputs.genus_bin_path, index=False)

    # ===== Time series summary =====
    vel = grouped.loc[grouped["velocity_km_per_myr"].notna(), ["time_bin", "velocity_km_per_myr"]].copy()
    if len(vel) == 0:
        raise RuntimeError("No valid velocity transitions found; check filters/inputs.")

    outlier_cap = float(vel["velocity_km_per_myr"].quantile(outlier_velocity_quantile))
    vel = vel[vel["velocity_km_per_myr"] <= outlier_cap]

    ts = (
        vel.groupby("time_bin")["velocity_km_per_myr"]
        .agg(n="size", mean="mean", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
        .sort_values("time_bin", ascending=False)
    )
    ts["temperature_c"] = ts["time_bin"].map(_get_temperature_c)
    ts.to_csv(outputs.velocity_timeseries_path, index=False)

    # ===== Top movers table (genus-level) =====
    genus_vel = (
        grouped.loc[grouped["velocity_km_per_myr"].notna() & ~grouped["censored_at_youngest_bin"]]
        .groupby("genus")["velocity_km_per_myr"]
        .agg(n_transitions="size", median_velocity="median", mean_velocity="mean")
        .reset_index()
    )
    genus_vel = genus_vel[genus_vel["n_transitions"] >= 3].copy()
    genus_vel = genus_vel.sort_values("median_velocity", ascending=False).head(30)
    genus_vel.to_csv(outputs.top_movers_path, index=False)

    # ===== Terminal extinction association =====
    model_df = grouped.copy()
    model_df = model_df[~model_df["censored_at_youngest_bin"]].copy()
    model_df = model_df.dropna(
        subset=[
            "velocity_km_per_myr",
            "abundance",
            "geographic_range",
            "lat_range",
            "env_breadth",
            "age_bins",
        ]
    )
    model_df = model_df[model_df["velocity_km_per_myr"] <= outlier_cap].copy()

    feature_cols = ["velocity_km_per_myr", "abundance", "geographic_range", "lat_range", "env_breadth", "age_bins"]
    X = model_df[feature_cols].astype(float)
    y = model_df["terminal_bin"].astype(int)
    groups = model_df["genus"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    (train_idx, test_idx) = next(splitter.split(X, y, groups=groups))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    prob = model.predict_proba(X.iloc[test_idx])[:, 1]
    auc = float(roc_auc_score(y.iloc[test_idx], prob))

    coefs = model.named_steps["clf"].coef_.reshape(-1)
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef_per_1sd": coefs,
            "odds_ratio_per_1sd": np.exp(coefs),
        }
    ).sort_values("odds_ratio_per_1sd", ascending=False)
    coef_df.to_csv(outputs.model_coefficients_path, index=False)

    metrics = {
        "data_path": data_path,
        "time_bin_width_myr": time_bin_width_myr,
        "locality_bin_degrees": locality_bin_degrees,
        "max_delta_myr_for_velocity": max_delta_myr_for_velocity,
        "min_occurrences_per_genus_bin": min_occurrences_per_genus_bin,
        "outlier_velocity_quantile": outlier_velocity_quantile,
        "n_occurrence_rows_loaded": n_rows_loaded,
        "n_occurrence_rows_after_clean": n_rows_after_clean,
        "n_occurrence_rows_after_dedup": n_rows_after_dedup,
        "n_occurrence_rows_after_coords": n_rows_after_coords,
        "deduped_fraction_of_clean": float(1.0 - (n_rows_after_dedup / n_rows_after_clean)) if n_rows_after_clean else 0.0,
        "youngest_time_bin_ma": youngest_bin,
        "n_genus_bin_rows": int(len(grouped)),
        "n_velocity_rows": int(grouped["velocity_km_per_myr"].notna().sum()),
        "n_model_rows": int(len(model_df)),
        "terminal_rate_model_rows": float(y.mean()) if len(y) else 0.0,
        "auc_group_split": auc,
    }

    # Repeated group splits for more stable performance/effect summaries.
    if n_model_repeats > 1 and len(model_df) > 0:
        rows = []
        for i in range(int(n_model_repeats)):
            splitter_i = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state + i)
            (tr_i, te_i) = next(splitter_i.split(X, y, groups=groups))

            model_i = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=random_state + i,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            model_i.fit(X.iloc[tr_i], y.iloc[tr_i])
            prob_i = model_i.predict_proba(X.iloc[te_i])[:, 1]
            auc_i = float(roc_auc_score(y.iloc[te_i], prob_i))
            coef_i = model_i.named_steps["clf"].coef_.reshape(-1)

            row = {"repeat": i, "auc": auc_i}
            for feature_name, coef_val in zip(feature_cols, coef_i, strict=True):
                row[f"coef_{feature_name}"] = float(coef_val)
                row[f"or_{feature_name}"] = float(np.exp(coef_val))
            rows.append(row)

        repeats_df = pd.DataFrame(rows)
        repeats_df.to_csv(outputs.model_repeats_path, index=False)

        metrics["auc_repeats_mean"] = float(repeats_df["auc"].mean())
        metrics["auc_repeats_sd"] = float(repeats_df["auc"].std(ddof=1)) if len(repeats_df) > 1 else 0.0
        metrics["auc_repeats_p2_5"] = float(repeats_df["auc"].quantile(0.025))
        metrics["auc_repeats_p97_5"] = float(repeats_df["auc"].quantile(0.975))

        summary_rows = []
        for feature_name in feature_cols:
            coef_col = f"coef_{feature_name}"
            or_col = f"or_{feature_name}"
            summary_rows.append(
                {
                    "feature": feature_name,
                    "coef_mean": float(repeats_df[coef_col].mean()),
                    "coef_p2_5": float(repeats_df[coef_col].quantile(0.025)),
                    "coef_p97_5": float(repeats_df[coef_col].quantile(0.975)),
                    "odds_ratio_mean": float(repeats_df[or_col].mean()),
                    "odds_ratio_p2_5": float(repeats_df[or_col].quantile(0.025)),
                    "odds_ratio_p97_5": float(repeats_df[or_col].quantile(0.975)),
                }
            )
        pd.DataFrame(summary_rows).sort_values("odds_ratio_mean", ascending=False).to_csv(
            outputs.model_coefficients_summary_path, index=False
        )

    outputs.model_metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    # ===== Figures =====
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: velocity time series with IQR and temperature overlay (secondary axis).
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(ts["time_bin"], ts["median"], color="#1f77b4", linewidth=2, label="Median velocity")
    ax1.fill_between(ts["time_bin"], ts["q25"], ts["q75"], color="#1f77b4", alpha=0.2, label="IQR")
    ax1.invert_xaxis()
    ax1.set_xlabel("Time (Ma)")
    ax1.set_ylabel("Paleo-velocity (km / Myr)")
    ax1.set_title("Genus paleo-velocity through time (PBDB interval-limited sample)")

    ax2 = ax1.twinx()
    ax2.plot(ts["time_bin"], ts["temperature_c"], color="#d62728", linewidth=1.5, alpha=0.85, label="Temp (°C)")
    ax2.set_ylabel("Approx. global temperature (°C)")

    # Mass extinction markers (approx. boundary ages).
    for age, label in [
        (444, "End-Ordovician"),
        (372, "Late Devonian"),
        (252, "End-Permian"),
        (201, "End-Triassic"),
        (66, "End-Cretaceous"),
    ]:
        if ts["time_bin"].min() <= age <= ts["time_bin"].max():
            ax1.axvline(age, color="black", alpha=0.2, linewidth=1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(outputs.fig_velocity_timeseries_path, dpi=200)
    plt.close(fig)

    # Figure 2: velocity at terminal vs non-terminal bins (within interval; censored genera removed).
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = model_df[["terminal_bin", "velocity_km_per_myr"]].copy()
    # Subsample non-terminal for readability.
    non_terminal = plot_df[plot_df["terminal_bin"] == 0]
    terminal = plot_df[plot_df["terminal_bin"] == 1]
    if len(non_terminal) > 50_000:
        non_terminal = non_terminal.sample(n=50_000, random_state=random_state)
    if len(terminal) > 50_000:
        terminal = terminal.sample(n=50_000, random_state=random_state)

    ax.boxplot(
        [non_terminal["velocity_km_per_myr"], terminal["velocity_km_per_myr"]],
        tick_labels=["Non-terminal bins", "Terminal bins"],
        showfliers=False,
    )
    ax.set_ylabel("Paleo-velocity (km / Myr)")
    ax.set_title("Mobility differences at terminal vs non-terminal genus occurrences")
    fig.tight_layout()
    fig.savefig(outputs.fig_terminal_vs_nont_path, dpi=200)
    plt.close(fig)

    return outputs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a genus paleo-velocity study and write paper-ready outputs.")
    p.add_argument("--data", default="data/processed/merged_occurrences.parquet", help="Input parquet file")
    p.add_argument("--out", default="thesis/archive/paleovelocity_pilot/output", help="Output directory")
    p.add_argument("--bin-width", type=float, default=5.0, help="Time bin width (Myr)")
    p.add_argument("--locality-bin", type=float, default=5.0, help="Locality grid size (degrees)")
    p.add_argument("--max-delta", type=float, default=10.0, help="Max time gap (Myr) to compute velocity")
    p.add_argument("--min-occ", type=int, default=1, help="Min occurrences per genus-bin to include")
    p.add_argument("--outlier-q", type=float, default=0.995, help="Velocity outlier cap quantile")
    p.add_argument("--repeats", type=int, default=25, help="Repeated group splits for model stability")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outs = run_paleovelocity_study(
        data_path=args.data,
        output_dir=args.out,
        time_bin_width_myr=args.bin_width,
        locality_bin_degrees=args.locality_bin,
        max_delta_myr_for_velocity=args.max_delta,
        min_occurrences_per_genus_bin=args.min_occ,
        outlier_velocity_quantile=args.outlier_q,
        n_model_repeats=args.repeats,
        random_state=args.seed,
    )
    print(f"Wrote outputs under: {outs.output_dir}")
