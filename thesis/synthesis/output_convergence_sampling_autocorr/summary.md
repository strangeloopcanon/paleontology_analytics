# Robustness: convergence vs volatility with sampling + autocorrelation-aware tests

We merge:
- Convergence bins: `thesis/convergence/output_v2/timebin_metrics.csv`
- Extended PBDB occurrences (for sampling proxies): `data/processed/pbdb_occurrences_extended.parquet`
- Independent forcing: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`

Volatility predictor: `delta_from_prev_T_field_meanabs` (CESM).
Convergence outcome: `functional_excess_similarity_js` (PBDB ecospace v2).

## Partial correlation tests

- IID permutation p-values shuffle residuals (exchangeable bins).
- Circular-shift p-values preserve autocorrelation structure (time-ordered bins).

- control_time: iid(corr=0.496, p=0.00405, n=33); shift(corr=0.496, p=5e-05, n=33)
- control_time_loc: iid(corr=0.351, p=0.0433, n=33); shift(corr=0.351, p=0.0319, n=33)
- control_time_loc_coll_occ: iid(corr=0.418, p=0.0154, n=33); shift(corr=0.418, p=0.032, n=33)
- control_time_loc_coll_occ_prov: iid(corr=0.405, p=0.0209, n=33); shift(corr=0.405, p=0.0328, n=33)

## Outputs

- Merged table: `thesis/synthesis/output_convergence_sampling_autocorr/merged.csv`
- Stats: `thesis/synthesis/output_convergence_sampling_autocorr/analysis_results.json`
- Figures: `thesis/synthesis/output_convergence_sampling_autocorr/figures`

## Notes

- Sampling proxies are derived from PBDB `collection_no` and a coarse environment classifier on PBDB `environment` strings; treat as approximate.
- For final inference, prefer explicit time-series models or block bootstraps, and integrate Macrostrat/rock-area covariates.

