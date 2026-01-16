# PBDB ingestion (local data build)

This folder contains scripts to download Paleobiology Database (PBDB) occurrence data in a paged/restartable way and build the local
parquet tables used throughout `thesis/`.

## Outputs (gitignored)

- Raw CSV downloads: `data/raw/`
- Processed parquets: `data/processed/` (canonical, extended, merged)

## Reproduce (typical)

1) Download a PBDB slice as a CSV (defaults target the Paleogene→Holocene slice used to fill the Cenozoic):

`python thesis/pbdb/download_pbdb_occurrences_paged.py --out data/raw/pbdb_occurrences_paleogene_holocene_paged.csv`

2) Build the canonical and extended parquets from local PBDB CSVs:

`python thesis/pbdb/build_pbdb_canonical_parquet.py --pattern 'data/raw/pbdb_occurrences_*.csv' --out data/processed/pbdb_occurrences.parquet`

`python thesis/pbdb/build_pbdb_extended_parquet.py --pattern 'data/raw/pbdb_occurrences_*.csv' --out data/processed/pbdb_occurrences_extended.parquet`

Notes:
- These scripts intentionally keep outputs local-only (large + regeneratable).
- For the full “one button” research pipeline once `data/processed/merged_occurrences.parquet` exists, see `thesis/run_all.py`.
