from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis.paleobiotic_velocity.run_pipeline import load_occurrences  # noqa: E402


CoordsMode = Literal["paleo", "modern"]


@dataclass(frozen=True)
class Event:
    name: str
    boundary_ma: float


DEFAULT_EVENTS: list[Event] = [
    Event("end_ordovician", 444.0),
    Event("late_devonian", 372.0),
    Event("end_permian", 252.0),
    Event("end_triassic", 201.0),
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _canonicalize_lng(lng_deg: float) -> float:
    # Map to [-180, 180) so 180° becomes -180° for grid wrap consistency.
    return float(((lng_deg + 180.0) % 360.0) - 180.0)


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


def _circular_span_deg(longitudes_deg: np.ndarray) -> float:
    if len(longitudes_deg) == 0:
        return float("nan")
    if len(longitudes_deg) == 1:
        return 0.0

    vals = ((longitudes_deg.astype(float) + 360.0) % 360.0).copy()
    vals.sort()
    gaps = np.diff(vals)
    wrap_gap = (vals[0] + 360.0) - vals[-1]
    max_gap = float(max(float(np.max(gaps)), float(wrap_gap)))
    return float(360.0 - max_gap)


def _connected_component_sizes(
    lat_bins: np.ndarray, lng_bins: np.ndarray, *, grid_deg: float
) -> tuple[list[int], int]:
    # Build integer grid indices so we can wrap longitude cleanly.
    if len(lat_bins) == 0:
        return [], 0

    n_lon = int(round(360.0 / float(grid_deg)))
    nodes: set[tuple[int, int]] = set()
    for la, lo in zip(lat_bins, lng_bins, strict=True):
        lo = _canonicalize_lng(float(lo))
        lat_i = int(round(float(la) / float(grid_deg)))
        lon_i = int(round((lo + 180.0) / float(grid_deg))) % n_lon
        nodes.add((lat_i, lon_i))

    if not nodes:
        return [], 0

    visited: set[tuple[int, int]] = set()
    comp_sizes: list[int] = []
    for node in nodes:
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        size = 0
        while stack:
            lat_i, lon_i = stack.pop()
            size += 1
            neighbors = [
                (lat_i - 1, lon_i),
                (lat_i + 1, lon_i),
                (lat_i, (lon_i - 1) % n_lon),
                (lat_i, (lon_i + 1) % n_lon),
            ]
            for nb in neighbors:
                if nb in nodes and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comp_sizes.append(size)

    return comp_sizes, len(nodes)


def _connectedness_metrics(
    lat_bins: np.ndarray, lng_bins: np.ndarray, *, grid_deg: float
) -> dict[str, float]:
    comp_sizes, n_nodes = _connected_component_sizes(lat_bins, lng_bins, grid_deg=grid_deg)
    if n_nodes <= 0 or not comp_sizes:
        return {
            "n_components": 0.0,
            "largest_component_frac": float("nan"),
            "component_entropy": float("nan"),
            "effective_components": float("nan"),
            "component_evenness": float("nan"),
        }

    n_components = len(comp_sizes)
    largest_frac = float(max(comp_sizes) / n_nodes)
    p = np.asarray(comp_sizes, dtype=float) / float(n_nodes)
    entropy = float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    effective = float(np.exp(entropy))
    evenness = float(entropy / np.log(n_components)) if n_components > 1 else 0.0
    return {
        "n_components": float(n_components),
        "largest_component_frac": largest_frac,
        "component_entropy": entropy,
        "effective_components": effective,
        "component_evenness": evenness,
    }


def _circular_mean_deg(longitudes_deg: np.ndarray) -> float:
    rad = np.deg2rad(longitudes_deg.astype(float))
    s = np.nanmean(np.sin(rad))
    c = np.nanmean(np.cos(rad))
    return float(np.rad2deg(np.arctan2(s, c)))


def compute_pre_event_metrics(pre_df: pd.DataFrame, *, grid_deg: float) -> pd.DataFrame:
    # Base metrics.
    base = (
        pre_df.groupby("genus")
        .agg(
            abundance=("genus", "size"),
            geographic_range=("locality", "nunique"),
            lat_min=("analysis_lat", "min"),
            lat_max=("analysis_lat", "max"),
            env_breadth=("environment", "nunique"),
            centroid_lat=("analysis_lat", "median"),
            centroid_lng=("analysis_lng", lambda s: _circular_mean_deg(s.to_numpy())),
            phylum=("phylum", lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA),
        )
        .reset_index()
    )
    base["lat_range"] = base["lat_max"] - base["lat_min"]
    base["cross_equator"] = ((base["lat_min"] < 0) & (base["lat_max"] > 0)).astype(int)
    base["centroid_abs_lat"] = base["centroid_lat"].abs()
    base["phylum"] = base["phylum"].fillna("NO_PHYLUM_SPECIFIED")
    centroid_map = base.set_index("genus")[["centroid_lat", "centroid_lng"]]

    # Connectedness/fragmentation metrics from unique locality bins.
    uniq = pre_df.drop_duplicates(subset=["genus", "lat_bin", "lng_bin"])[["genus", "lat_bin", "lng_bin"]]
    rows = []
    for genus, g in uniq.groupby("genus", sort=False):
        metrics = _connectedness_metrics(
            g["lat_bin"].to_numpy(dtype=float), g["lng_bin"].to_numpy(dtype=float), grid_deg=grid_deg
        )

        # Dispersion: mean distance from occupied cells to the genus centroid.
        # Uses grid-cell centers (lat_bin/lng_bin) to reduce oversampling sensitivity.
        centroid = centroid_map.loc[genus]
        dists = [
            _haversine_km(float(la), float(lo), float(centroid["centroid_lat"]), float(centroid["centroid_lng"]))
            for la, lo in zip(g["lat_bin"].to_numpy(dtype=float), g["lng_bin"].to_numpy(dtype=float), strict=True)
        ]
        dispersion_km = float(np.mean(dists)) if dists else float("nan")
        lon_span = _circular_span_deg(g["lng_bin"].to_numpy(dtype=float))

        rows.append(
            {
                "genus": genus,
                "n_components": metrics["n_components"],
                "largest_component_frac": metrics["largest_component_frac"],
                "component_entropy": metrics["component_entropy"],
                "effective_components": metrics["effective_components"],
                "component_evenness": metrics["component_evenness"],
                "dispersion_km": dispersion_km,
                "lon_span_deg": float(lon_span),
            }
        )
    frag = pd.DataFrame(rows)

    out = base.merge(frag, on="genus", how="left")
    out["component_density"] = out["n_components"] / out["geographic_range"].replace(0, np.nan)

    # Log transforms (avoid dominating by extremely large genera).
    out["log_abundance"] = np.log1p(out["abundance"].astype(float))
    out["log_geographic_range"] = np.log1p(out["geographic_range"].astype(float))
    out["log_env_breadth"] = np.log1p(out["env_breadth"].astype(float))
    out["log_dispersion_km"] = np.log1p(out["dispersion_km"].astype(float))
    out["log_n_components"] = np.log1p(out["n_components"].astype(float))
    out["log_effective_components"] = np.log(out["effective_components"].astype(float).clip(lower=1.0))
    return out


def _event_bins(boundary_ma: float, *, time_bin_myr: float) -> tuple[float, float]:
    pre_bin = float(np.ceil(boundary_ma / time_bin_myr) * time_bin_myr)
    post_bin = float(pre_bin - time_bin_myr)
    return pre_bin, post_bin


def build_event_dataset(
    df: pd.DataFrame,
    *,
    boundary_ma: float,
    time_bin_myr: float,
    grid_deg: float,
    post_window_myr: float,
) -> pd.DataFrame:
    pre_bin, post_bin = _event_bins(boundary_ma, time_bin_myr=time_bin_myr)

    pre = df[df["time_bin"] == pre_bin].copy()
    if pre.empty:
        raise ValueError(f"No data found for pre-event bin {pre_bin} Ma (boundary={boundary_ma} Ma)")

    metrics = compute_pre_event_metrics(pre, grid_deg=grid_deg)

    # Survivorship definitions.
    youngest_bin = df.groupby("genus")["time_bin"].min()
    metrics["survived_any"] = (youngest_bin.reindex(metrics["genus"]).to_numpy() <= post_bin).astype(int)

    n_bins = int(round(post_window_myr / time_bin_myr))
    post_bins = [float(post_bin - i * time_bin_myr) for i in range(n_bins + 1)]
    genera_in_window = set(df[df["time_bin"].isin(post_bins)]["genus"].unique())
    metrics["survived_10myr"] = metrics["genus"].isin(genera_in_window).astype(int)

    metrics["event_boundary_ma"] = float(boundary_ma)
    metrics["pre_bin_ma"] = float(pre_bin)
    metrics["post_bin_ma"] = float(post_bin)
    return metrics


def fit_repeated_logit(
    data: pd.DataFrame,
    *,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    repeats: int,
    seed: int,
) -> dict:
    df = data.dropna(subset=numeric_features + categorical_features + [target]).copy()
    X_num = df[numeric_features].astype(float)
    y = df[target].astype(int).to_numpy()

    transformers = [("num", StandardScaler(), numeric_features)]
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    splitter = StratifiedShuffleSplit(n_splits=repeats, test_size=0.3, random_state=seed)
    rows = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(X_num, y)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        pipe.fit(train_df[numeric_features + categorical_features], train_df[target])
        prob = pipe.predict_proba(test_df[numeric_features + categorical_features])[:, 1]
        auc = float(roc_auc_score(test_df[target], prob))
        ap = float(average_precision_score(test_df[target], prob))

        coef = pipe.named_steps["clf"].coef_.reshape(-1)
        num_coef = coef[: len(numeric_features)]
        row = {"repeat": i, "auc": auc, "avg_precision": ap}
        for name, c in zip(numeric_features, num_coef, strict=True):
            row[f"coef_{name}"] = float(c)
            row[f"or_{name}"] = float(np.exp(c))
        rows.append(row)

    rep = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(df)),
        "event_rate": float(df[target].mean()),
        "auc_mean": float(rep["auc"].mean()),
        "auc_sd": float(rep["auc"].std(ddof=1)) if len(rep) > 1 else 0.0,
        "ap_mean": float(rep["avg_precision"].mean()),
        "ap_sd": float(rep["avg_precision"].std(ddof=1)) if len(rep) > 1 else 0.0,
    }

    coef_rows = []
    for name in numeric_features:
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


def _plot_or(coef_summary: pd.DataFrame, *, title: str, out_path: Path) -> None:
    if coef_summary.empty:
        return
    plot = coef_summary.set_index("feature").loc[::-1]
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
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--out", default="thesis/geographic_portfolio/output")
    p.add_argument("--coords-mode", choices=["paleo", "modern", "both"], default="both")
    p.add_argument("--time-bin-myr", type=float, default=5.0)
    p.add_argument("--grid-deg", type=float, default=5.0)
    p.add_argument("--post-window-myr", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-phylum", action="store_true", help="Include phylum fixed effects (one-hot).")
    args = p.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    modes: list[CoordsMode]
    if args.coords_mode == "both":
        modes = ["paleo", "modern"]
    else:
        modes = [args.coords_mode]  # type: ignore[assignment]

    summary_rows: list[dict] = []

    for mode in modes:
        print(f"Loading occurrences (coords_mode={mode})…")
        df = load_occurrences(
            args.data,
            coords_mode=mode,  # type: ignore[arg-type]
            time_bin_myr=float(args.time_bin_myr),
            locality_bin_deg=float(args.grid_deg),
        )

        mode_dir = out_dir / mode
        _ensure_dir(mode_dir)
        _ensure_dir(mode_dir / "datasets")
        _ensure_dir(mode_dir / "results")
        _ensure_dir(mode_dir / "figures")

        for ev in DEFAULT_EVENTS:
            print(f"Event: {ev.name} (boundary={ev.boundary_ma} Ma)")
            ds = build_event_dataset(
                df,
                boundary_ma=ev.boundary_ma,
                time_bin_myr=float(args.time_bin_myr),
                grid_deg=float(args.grid_deg),
                post_window_myr=float(args.post_window_myr),
            )

            for target in ["survived_any", "survived_10myr"]:
                dataset_path = mode_dir / "datasets" / f"{ev.name}_{target}_dataset.parquet"
                ds.to_parquet(dataset_path, index=False)

                numeric = [
                    "log_abundance",
                    "log_geographic_range",
                    "lat_range",
                    "log_env_breadth",
                    "largest_component_frac",
                ]
                categorical = ["phylum"] if args.with_phylum else []

                fit_full = fit_repeated_logit(
                    ds,
                    target=target,
                    numeric_features=numeric,
                    categorical_features=categorical,
                    repeats=int(args.repeats),
                    seed=int(args.seed),
                )

                baseline_numeric = [c for c in numeric if c != "largest_component_frac"]
                fit_base = fit_repeated_logit(
                    ds,
                    target=target,
                    numeric_features=baseline_numeric,
                    categorical_features=categorical,
                    repeats=int(args.repeats),
                    seed=int(args.seed),
                )

                coef_path = mode_dir / "results" / f"{ev.name}_{target}_coef_summary.csv"
                fit_full["coef_summary"].to_csv(coef_path, index=False)
                rep_path = mode_dir / "results" / f"{ev.name}_{target}_repeats.csv"
                fit_full["repeats"].to_csv(rep_path, index=False)

                meta = {
                    "coords_mode": mode,
                    "event": ev.__dict__,
                    "target": target,
                    "time_bin_myr": float(args.time_bin_myr),
                    "grid_deg": float(args.grid_deg),
                    "post_window_myr": float(args.post_window_myr),
                    "with_phylum": bool(args.with_phylum),
                    "full_model": fit_full["summary"],
                    "baseline_model": fit_base["summary"],
                    "delta_auc_full_minus_baseline": float(
                        fit_full["summary"]["auc_mean"] - fit_base["summary"]["auc_mean"]
                    ),
                }
                meta_path = mode_dir / "results" / f"{ev.name}_{target}_meta.json"
                meta_path.write_text(json.dumps(meta, indent=2) + "\n")

                fig_path = mode_dir / "figures" / f"{ev.name}_{target}_odds_ratios.png"
                _plot_or(
                    fit_full["coef_summary"],
                    title=f"{ev.name} ({mode}) – {target}",
                    out_path=fig_path,
                )

                # Summary row (focus on connectedness feature).
                coef = fit_full["coef_summary"].set_index("feature")
                lcf = coef.loc["largest_component_frac"].to_dict() if "largest_component_frac" in coef.index else {}
                summary_rows.append(
                    {
                        "coords_mode": mode,
                        "event": ev.name,
                        "boundary_ma": ev.boundary_ma,
                        "target": target,
                        "n": fit_full["summary"]["n_rows"],
                        "event_rate": fit_full["summary"]["event_rate"],
                        "auc_full": fit_full["summary"]["auc_mean"],
                        "auc_base": fit_base["summary"]["auc_mean"],
                        "delta_auc": meta["delta_auc_full_minus_baseline"],
                        "or_largest_component_frac": lcf.get("odds_ratio_mean", np.nan),
                        "or_largest_component_frac_p2_5": lcf.get("odds_ratio_p2_5", np.nan),
                        "or_largest_component_frac_p97_5": lcf.get("odds_ratio_p97_5", np.nan),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
