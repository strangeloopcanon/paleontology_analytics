# Paleovelocity paper (draft + reproducible pipeline)

This folder contains a manuscript draft plus reproducible outputs for a *genus paleo-velocity* study, inferred from PBDB paleocoordinate shifts through time.

## Reproduce

From the repo root:

```bash
python -m src.analysis.paleovelocity \
  --data data/processed/merged_occurrences.parquet \
  --out paper/paleovelocity
```

Outputs:
- `paper/paleovelocity/figures/` (`fig1_velocity_timeseries.png`, `fig2_terminal_vs_nont_velocity.png`)
- `paper/paleovelocity/results/` (model metrics, coefficients, time series, full genus-bin feature table)
- `paper/paleovelocity/tables/` (`top_movers.csv`)

## Data

This analysis expects `data/processed/merged_occurrences.parquet` to exist. If you need to regenerate it, the repo includes a pipeline:

```bash
python -m src.cli download --source pbdb --interval Cambrian,Cretaceous --output data/raw
python -m src.cli normalize --source pbdb --input data/raw --output data/processed
python -m src.cli download --source neotoma --output data/raw
python -m src.cli normalize --source neotoma --input data/raw --output data/processed
python -m src.cli normalize --source merge --output data/processed
```

Notes:
- The current PBDB interval setting (`Cambrian,Cretaceous`) truncates the record at ~66 Ma, so genera whose last appearance is at the youngest bin are treated as *right-censored* in the extinction analysis.
- `data/raw/` and `data/processed/` are gitignored by default.
