# Robustness: convergence vs volatility with sampling + autocorrelation-aware tests

We merge:
- Convergence bins: `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`
- Extended PBDB occurrences (for sampling proxies): `data/processed/pbdb_occurrences_extended.parquet`
- Independent forcing: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`
- Macrostrat proxies: `data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv`

Volatility predictor: `delta_from_prev_T_field_meanabs` (CESM).
Convergence outcome: `functional_excess_similarity_js` (PBDB ecospace v2).

## Partial correlation tests

- IID permutation p-values shuffle residuals (exchangeable bins).
- Circular-shift p-values preserve autocorrelation structure (time-ordered bins).

- control_time: iid(corr=0.409, p=0.00825, n=40); shift(corr=0.409, p=0.0263, n=40)
- control_time_loc: iid(corr=0.289, p=0.0706, n=40); shift(corr=0.289, p=0.106, n=40)
- control_time_loc_coll_occ: iid(corr=0.350, p=0.0273, n=40); shift(corr=0.350, p=0.0265, n=40)
- control_time_loc_coll_occ_prov: iid(corr=0.358, p=0.0247, n=40); shift(corr=0.358, p=0.0283, n=40)
- control_time_loc_coll_occ_prov_macrostrat: iid(corr=0.241, p=0.133, n=40); shift(corr=0.241, p=0.129, n=40)

## Outputs

- Merged table: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat/merged.csv`
- Stats: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat/analysis_results.json`
- Figures: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat/figures`

## Notes

- Sampling proxies are derived from PBDB `collection_no` and a coarse environment classifier on PBDB `environment` strings; treat as approximate.
- For final inference, prefer explicit time-series models or block bootstraps, and integrate Macrostrat/rock-area covariates.

