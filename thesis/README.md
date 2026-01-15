# Research workspace

## Contents

- `pilot/`: the initial paleovelocity pilot migrated from the repo root (code + manuscript + outputs).
- `literature/`: reproducible bibliography build + annotated review notes.
- `paleobiotic_velocity/`: the rigorous pipeline (bias controls + survival models + robustness tests) and its outputs.
- `geographic_portfolio/`: event-based survivorship analysis testing range configuration (“connectedness/portfolio”) across mass extinctions + manuscript draft.
- `convergence/`: marine functional convergence across provinces using PBDB ecospace roles.
- `earth_system/`: independent CESM-derived forcing series (Li et al. 2022) + derived coherence/patchiness metrics.
- `macrostrat/`: rock-record proxy ingestion and binned time series.
- `synthesis/`: end-to-end robustness, pair-level model, interpretability, and publication-grade inference checks.
- `manuscript_convergence_volatility/`: draft paper + supplement focused on volatility → marine functional convergence.
- `figures/`: a small, curated set of figures suitable for embedding in READMEs/writeups.

## Tracked vs generated

- Tracked: writeups (`*.md`), analysis code (`*.py`), curated figures (`figures/`), and selected open-access PDFs (`literature/pdfs/`).
- Not tracked: large intermediate artifacts (raw tables, caches) produced under `output*/`, plus large literature query exports (regeneratable).

## Reproduce key results

Run the “one button” pipeline (uses existing processed data; no large downloads):

`python thesis/run_all.py`

## Highlights

### Marine functional convergence vs climate volatility

![Marine volatility vs convergence](figures/marine_volatility_vs_convergence.png)

### Pair-level model (publication-oriented)

![Pair-level volatility model](figures/pair_level_volatility_model.png)

### Bin-level time-series model fit

![Bin-level time series fit](figures/bin_level_time_series_fit.png)
