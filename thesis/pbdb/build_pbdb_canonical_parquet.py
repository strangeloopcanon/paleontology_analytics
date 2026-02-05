from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.normalization.schema import OCCURRENCE_SCHEMA, PBDB_MAPPING  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_files(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    return [f for f in files if Path(f).is_file()]


def _finalize_canonical(df: pd.DataFrame) -> pd.DataFrame:
    for col in OCCURRENCE_SCHEMA.keys():
        if col not in df.columns:
            df[col] = None

    df = df[list(OCCURRENCE_SCHEMA.keys())].copy()

    for col, dtype in OCCURRENCE_SCHEMA.items():
        if dtype == "string":
            s = df[col].astype("string").str.strip()
            df[col] = s.mask(s == "", pd.NA)
        elif dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Unsupported dtype in schema: {dtype}")
    return df


def _normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    df = chunk.rename(columns=PBDB_MAPPING).copy()
    df["source_db"] = "PBDB"
    if "mid_ma" not in df.columns:
        df["max_ma"] = pd.to_numeric(df.get("max_ma"), errors="coerce")
        df["min_ma"] = pd.to_numeric(df.get("min_ma"), errors="coerce")
        df["mid_ma"] = (df["max_ma"] + df["min_ma"]) / 2.0
    return _finalize_canonical(df)


def build_pbdb_canonical_parquet(
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

    needed_cols = set(PBDB_MAPPING.keys())
    for f in files:
        print(f"Reading: {f}")
        reader = pd.read_csv(
            f,
            usecols=lambda c: c in needed_cols,
            chunksize=int(chunksize),
            low_memory=False,
        )
        for chunk in reader:
            if "occurrence_no" not in chunk.columns:
                continue

            if deduplicate:
                occ = pd.to_numeric(chunk["occurrence_no"], errors="coerce")
                keep = occ.notna()
                chunk = chunk.loc[keep].copy()
                occ = occ.loc[keep].astype("int64")
                if seen_occ:
                    keep2 = ~occ.isin(seen_occ)
                    if not bool(keep2.all()):
                        chunk = chunk.loc[keep2].copy()
                        occ = occ.loc[keep2]
                seen_occ.update(map(int, occ.tolist()))

            norm = _normalize_chunk(chunk)
            total_rows += int(len(norm))
            table = pa.Table.from_pandas(norm, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
            writer.write_table(table)

    if writer is not None:
        writer.close()
    print(f"Wrote {total_rows:,} rows to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="data/raw/pbdb_occurrences_*.csv")
    p.add_argument("--out", default="data/processed/pbdb_occurrences.parquet")
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()

    files = _iter_files(str(args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {args.pattern}")

    build_pbdb_canonical_parquet(
        files,
        out_path=Path(args.out),
        chunksize=int(args.chunksize),
        deduplicate=not bool(args.no_dedup),
    )


if __name__ == "__main__":
    main()
