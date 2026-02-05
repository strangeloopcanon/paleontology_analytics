from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_timeseries(
    *,
    sections_csv: Path,
    out_csv: Path,
    time_bin_myr: float,
    max_time_ma: float,
) -> Path:
    sections_csv = Path(sections_csv)
    if not sections_csv.exists():
        raise FileNotFoundError(f"Missing Macrostrat sections CSV: {sections_csv}")

    df = pd.read_csv(sections_csv)
    for c in ["t_age", "b_age", "col_area", "col_id", "section_id", "project_id", "pbdb_collections"]:
        if c not in df.columns:
            raise ValueError(f"Macrostrat CSV missing expected column: {c}")

    df["t_age"] = pd.to_numeric(df["t_age"], errors="coerce")
    df["b_age"] = pd.to_numeric(df["b_age"], errors="coerce")
    df["col_area"] = pd.to_numeric(df["col_area"], errors="coerce")
    df["pbdb_collections"] = pd.to_numeric(df["pbdb_collections"], errors="coerce")
    df["col_id"] = pd.to_numeric(df["col_id"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["t_age", "b_age", "col_area"]).copy()
    df = df[df["t_age"] >= 0].copy()
    df = df[df["b_age"] >= 0].copy()
    df = df[df["t_age"] <= float(max_time_ma)].copy()
    df = df[df["b_age"] <= float(max_time_ma)].copy()

    df["mid_ma"] = (df["t_age"] + df["b_age"]) / 2.0
    df["time_bin"] = (df["mid_ma"] / float(time_bin_myr)).round() * float(time_bin_myr)

    # Per-bin totals.
    by_bin = df.groupby("time_bin", as_index=False).agg(
        macro_n_sections=("section_id", "count"),
        macro_n_columns=("col_id", lambda s: int(pd.Series(s).dropna().nunique())),
        macro_pbdb_collections_sum=("pbdb_collections", "sum"),
    )

    # Sum col_area across unique columns per bin (avoid double-counting if it happens).
    uniq_cols = df.drop_duplicates(subset=["time_bin", "col_id"])[["time_bin", "col_id", "col_area"]]
    area = uniq_cols.groupby("time_bin", as_index=False).agg(macro_col_area_sum=("col_area", "sum"))

    out = by_bin.merge(area, on="time_bin", how="left").sort_values("time_bin", ascending=False).reset_index(drop=True)

    out_csv = Path(out_csv)
    _ensure_dir(out_csv.parent)
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(out):,} bins)")
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sections", default="data/raw/external/macrostrat/sections_all.csv")
    p.add_argument("--out", default="data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--max-time-ma", type=float, default=540.0)
    args = p.parse_args()

    build_timeseries(
        sections_csv=Path(args.sections),
        out_csv=Path(args.out),
        time_bin_myr=float(args.time_bin_myr),
        max_time_ma=float(args.max_time_ma),
    )


if __name__ == "__main__":
    main()

