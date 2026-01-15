# Robustness (occ-ecospace): convergence vs volatility with sampling + autocorrelation-aware tests

We merge:
- Convergence bins: `thesis/convergence/output_occ_ecospace_tetrapoda_freshwater_min3_pairs15/timebin_metrics.csv`
- PBDB occs export (sampling proxies): `data/raw/pbdb_occurrences_tetrapoda_ecospace_paged.csv`
- Independent forcing: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`

Taxon-environment filter for sampling proxies: `freshwater` (substring on `taxon_environment`).

Volatility predictor: `delta_from_prev_T_field_meanabs` (CESM).
Convergence outcome: `functional_excess_similarity_js` (occ-level ecospace JS residual).

## Partial correlation tests

- IID permutation p-values shuffle residuals (exchangeable bins).
- Circular-shift p-values preserve autocorrelation structure (time-ordered bins).

- control_time: iid(corr=0.584, p=0.0212, n=15); shift(corr=0.584, p=5e-05, n=15)
- control_time_loc: iid(corr=0.343, p=0.214, n=15); shift(corr=0.343, p=0.218, n=15)
- control_time_loc_coll_occ: iid(corr=0.293, p=0.284, n=15); shift(corr=0.293, p=0.36, n=15)
- control_time_loc_coll_occ_prov: iid(corr=0.248, p=0.371, n=15); shift(corr=0.248, p=0.359, n=15)

## Outputs

- Merged table: `thesis/synthesis/output_convergence_sampling_autocorr_tetrapoda_freshwater_v1/merged.csv`
- Stats: `thesis/synthesis/output_convergence_sampling_autocorr_tetrapoda_freshwater_v1/analysis_results.json`
- Figures: `thesis/synthesis/output_convergence_sampling_autocorr_tetrapoda_freshwater_v1/figures`

## Notes

- Sampling proxies (`n_occurrences`, `n_collections`) are computed from the same PBDB occs export used to compute convergence.
- For publication-grade inference, prefer pair-level or hierarchical models and explicit time-series error structures.

