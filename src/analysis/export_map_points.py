from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd

from src.analysis.geo import add_analysis_coordinates


def export_map_points(
    data_path: str = "data/processed/merged_occurrences.parquet",
    output_file: str = "dashboard/map_points.json",
    *,
    time_bin_width_ma: float = 10.0,
    max_points_per_bin: int = 500,
    min_points_per_bin: int = 200,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Export a compact set of occurrence points for an animated paleomap.

    - Uses `analysis_lat`/`analysis_lng` (paleocoords when available, else modern coords).
    - Bins by time to keep the animation stable.
    - Samples up to `max_points_per_bin` points per bin for size and speed.
    """
    cols = ["mid_ma", "paleolat", "paleolng", "lat", "lng"]
    df = pd.read_parquet(data_path, columns=[c for c in cols if c is not None])
    df["mid_ma"] = pd.to_numeric(df["mid_ma"], errors="coerce")
    df = df.dropna(subset=["mid_ma"])
    df = add_analysis_coordinates(df)
    df = df.dropna(subset=["analysis_lat", "analysis_lng"])

    df["time_bin"] = (df["mid_ma"] / time_bin_width_ma).round() * time_bin_width_ma

    points_by_bin: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"lat": [], "lng": []})
    bins = []

    for time_bin, group in df.groupby("time_bin"):
        n = int(len(group))
        if n < min_points_per_bin:
            continue
        bins.append(float(time_bin))

        n_sample = min(max_points_per_bin, n)
        sampled = group.sample(n=n_sample, random_state=random_seed + int(time_bin))
        points_by_bin[str(float(time_bin))]["lat"] = sampled["analysis_lat"].astype(float).tolist()
        points_by_bin[str(float(time_bin))]["lng"] = sampled["analysis_lng"].astype(float).tolist()

    bins_sorted = sorted(bins, reverse=True)

    payload: dict[str, Any] = {
        "bins": bins_sorted,
        "points": points_by_bin,
        "params": {
            "time_bin_width_ma": time_bin_width_ma,
            "max_points_per_bin": max_points_per_bin,
            "min_points_per_bin": min_points_per_bin,
            "coord_preference": "paleocoords_if_available_else_modern",
        },
        "note": "Points are sampled per time bin for fast browser rendering.",
    }

    with open(output_file, "w") as f:
        json.dump(payload, f)

    return payload


if __name__ == "__main__":
    export_map_points()

