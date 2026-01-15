from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import ParserError
import pyarrow as pa
import pyarrow.parquet as pq


PBDB_USECOLS = [
    "occurrence_no",
    "collection_no",
    "max_ma",
    "min_ma",
    "lat",
    "lng",
    "paleolat",
    "paleolng",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "environment",
    "reference_no",
    "primary_reference",
    "formation",
    "geological_group",
    "member",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_pbdb_csvs(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    return [f for f in files if Path(f).is_file()]


def _normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["occurrence_no"] = pd.to_numeric(out["occurrence_no"], errors="coerce").astype("Int64")
    out["collection_no"] = pd.to_numeric(out["collection_no"], errors="coerce").astype("Int64")
    out["max_ma"] = pd.to_numeric(out["max_ma"], errors="coerce")
    out["min_ma"] = pd.to_numeric(out["min_ma"], errors="coerce")
    out["mid_ma"] = (out["max_ma"] + out["min_ma"]) / 2.0

    for c in ["lat", "lng", "paleolat", "paleolng"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ["phylum", "class", "order", "family", "genus", "environment", "primary_reference", "formation", "geological_group", "member"]:
        if c in out.columns:
            s = out[c].astype("string").str.strip()
            out[c] = s.mask(s == "", pd.NA)

    out["reference_no"] = pd.to_numeric(out["reference_no"], errors="coerce").astype("Int64")
    out["source_db"] = "PBDB"

    keep = ["source_db"] + [c for c in PBDB_USECOLS if c in out.columns] + ["mid_ma"]
    out = out[keep].copy()
    return out


def build_extended_parquet(
    files: Iterable[str],
    *,
    out_path: Path,
    chunksize: int = 250_000,
    deduplicate: bool = True,
) -> None:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    if out_path.exists():
        out_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    seen_occ: set[int] = set()
    for f in files:
        print(f"Reading: {f}")
        try:
            reader = pd.read_csv(
                f,
                usecols=lambda c: c in set(PBDB_USECOLS) | {"max_ma", "min_ma", "lat", "lng", "paleolat", "paleolng"},
                chunksize=int(chunksize),
                low_memory=False,
            )
            for chunk in reader:
                norm = _normalize_chunk(chunk)
                if deduplicate:
                    norm = norm.dropna(subset=["occurrence_no"]).drop_duplicates(subset=["occurrence_no"]).copy()
                    if seen_occ:
                        mask = ~norm["occurrence_no"].astype("int64").isin(seen_occ)
                        if not bool(mask.all()):
                            norm = norm.loc[mask].copy()
                    seen_occ.update(map(int, norm["occurrence_no"].astype("int64").tolist()))
                total_rows += int(len(norm))
                table = pa.Table.from_pandas(norm, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
                writer.write_table(table)
        except ParserError as e:
            print(f"Warning: skipping unreadable/incomplete CSV {f}: {e}")
    if writer is not None:
        writer.close()
    print(f"Wrote {total_rows} rows to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="data/raw/pbdb_occurrences_*.csv")
    p.add_argument("--out", default="data/processed/pbdb_occurrences_extended.parquet")
    p.add_argument("--chunksize", type=int, default=250_000)
    args = p.parse_args()

    files = _iter_pbdb_csvs(args.pattern)
    if not files:
        raise SystemExit(f"No files matched: {args.pattern}")
    build_extended_parquet(files, out_path=Path(args.out), chunksize=int(args.chunksize))


if __name__ == "__main__":
    main()
