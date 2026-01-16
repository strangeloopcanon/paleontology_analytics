# Data (local-only by default)

This repo does not track large datasets. By default, `data/raw/` and `data/processed/` are gitignored and are expected to be generated
locally.

Common local inputs/outputs:
- PBDB downloads and parquet builds: see `thesis/pbdb/`.
- Macrostrat proxy time series (sampling sensitivity control): see `thesis/macrostrat/`.

Note: `data/analysis/` contains small, generated plots produced by the `src` analysis scripts. They are safe to delete/regenerate.
