# Paleobiotic velocity (rigorous pipeline)

This directory contains a *bias-aware* mobility + extinction analysis pipeline designed for a thesis-level standard:

- multiple mobility estimators (occurrence-weighted vs locality-weighted centroids)
- coordinate negative control (modern coords vs PBDB paleocoordinates)
- discrete-time survival (hazard) models with time-bin fixed effects
- robustness runs across binning/weighting choices

## Run

From the repo root:

```bash
python thesis/paleobiotic_velocity/run_pipeline.py \
  --data data/processed/merged_occurrences.parquet \
  --out thesis/paleobiotic_velocity/output
```

Outputs are written under `thesis/paleobiotic_velocity/output/` (figures, tables, model summaries).

