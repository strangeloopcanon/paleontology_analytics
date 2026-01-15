# Stored results (exploratory): convergence from PBDB ecospace + independent forcing

This document summarizes the current “stored artifacts” for the three convergence questions.

## Q1) Functional convergence without taxonomic convergence (post‑perturbations)

Operationalization (PBDB-only, first pass):
- “Perturbation” proxy: global genus turnover between adjacent 10 Myr bins (Jaccard-based).
- “Functional convergence”: **functional excess similarity** = residual of (functional similarity ~ taxonomic similarity) across region pairs within a time bin.

Result (PBDB turnover proxy):
- No positive support in the raw bin-level correlations; after controlling for the strong long-term time trend, turnover is weakly **negative** (marginal).

Files:
- `thesis/convergence/output_v3_fullpbdb/summary.md`
- `thesis/convergence/output_v3_fullpbdb/analysis_results.json`

## Q2) Convergence vs fragmentation / provinciality

Operationalization:
- Fragmentation/provinciality proxy: `provinciality = 1 - mean_taxonomic_similarity` across region pairs per bin.

Result:
- Raw correlation is small, but **after controlling for time** (partial correlation), provinciality is **positively associated** with functional convergence (JS residual):
  - `corr(provinciality, convergence_js | time)` ≈ `+0.324`, perm‑p ≈ `0.039` (`n=40` bins; full PBDB incl. Cenozoic).

Interpretation (guarded):
- When provinces are more taxonomically distinct, they can still look **more similar in functional composition than expected** → consistent with “repeated filling” of similar ecospace roles across separated regions.

## Q3) Convergence peaks during volatility (not stability)

PBDB-only volatility proxy (turnover) was not compelling; we therefore tested an **independent** volatility series:

Independent dataset:
- Li et al. (2022) CESM snapshot simulations (10 Myr sampling; 540 Ma → PI), downloaded from figshare `10.6084/m9.figshare.19920662.v1`.
- Derived volatility series: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`

Result (independent climate volatility):
- Convergence correlates **positively** with temperature volatility, and the association survives sampling controls and an autocorrelation-aware null:
  - See: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb/summary.md`

Interpretation (guarded):
- When climate shifts more strongly between adjacent 10 Myr snapshots, distant regions become **more similar in “ecospace role composition” than taxonomy alone would predict**.

Files:
- `thesis/convergence/output_independent_forcing/summary.md`
- `thesis/convergence/output_independent_forcing/analysis_results.json`
- `thesis/convergence/output_independent_forcing/merged_convergence_earthsystem.csv`

## Global meta-result: strong long-term trend

Across bins, functional excess similarity shows a **very strong monotonic trend with time** (older → higher convergence). This is statistically strong but could reflect:
- real macroevolutionary/ecological changes (expansion of functional design space through the Phanerozoic), and/or
- differences in sampling, ecospace annotation completeness, or data structure through time.

See:
- `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`

## Guardrails / why this is still exploratory

- PBDB ecospace annotations have missingness and curation heterogeneity across taxa.
- PBDB occurrences still embed sampling/rock/collection effects; these analyses do not yet include rock/collection proxies or hierarchical models.
- The “functional convergence” measure is correlation-based and depends on the chosen bin/grid/thresholds.
