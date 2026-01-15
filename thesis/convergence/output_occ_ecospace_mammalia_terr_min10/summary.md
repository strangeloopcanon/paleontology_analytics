# Convergence (occurrence-level ecospace): PBDB subset

- Input PBDB CSV: `data/raw/pbdb_occurrences_mammalia_ecospace_paged.csv`
- Taxon-environment filter: `terrestrial` (substring on `taxon_environment`)
- Bins written: 6
- Global fit R²: 0.492

## Outputs

- time bins: `thesis/convergence/output_occ_ecospace_mammalia_terr_min10/timebin_metrics.csv`
- pairwise sample: `thesis/convergence/output_occ_ecospace_mammalia_terr_min10/pairwise_sample.csv`
- meta: `thesis/convergence/output_occ_ecospace_mammalia_terr_min10/analysis_results.json`
- figures: `thesis/convergence/output_occ_ecospace_mammalia_terr_min10/figures`

## Notes

- This mirrors the marine pipeline but uses PBDB *occurrence-level* ecospace fields (`diet`, `motility`, `life_habit`) from `occs/list`.
- For forcing tests, merge `timebin_metrics.csv` with `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv` on `time_bin`.

