# Robustness: convergence vs volatility with sampling + autocorrelation-aware tests

We merge:
- Convergence bins: `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`
- Extended PBDB occurrences (for sampling proxies): `data/processed/pbdb_occurrences_extended.parquet`
- Independent forcing: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`
- Macrostrat proxies: `data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv`
- Sampling PCA features: log1p(n_localities), log1p(marine_n_collections), log1p(marine_n_occurrences), log1p(macro_col_area_sum), log1p(macro_n_sections)
- Sampling PCA PC1 explained variance: 0.695

Volatility predictor: `delta_from_prev_T_field_meanabs` (CESM).
Convergence outcome: `functional_excess_similarity_js` (PBDB ecospace v2).

## Partial correlation tests

- IID permutation p-values shuffle residuals (exchangeable bins).
- Circular-shift p-values preserve autocorrelation structure (time-ordered bins).

- control_time: iid(corr=0.409, p=0.00825, n=40); shift(corr=0.409, p=0.0263, n=40)
- control_time_loc: iid(corr=0.289, p=0.0706, n=40); shift(corr=0.289, p=0.106, n=40)
- control_time_loc_coll_occ: iid(corr=0.350, p=0.0273, n=40); shift(corr=0.350, p=0.0265, n=40)
- control_time_loc_coll_occ_prov: iid(corr=0.358, p=0.0247, n=40); shift(corr=0.358, p=0.0283, n=40)
- control_time_sampling_pc1: iid(corr=0.401, p=0.00985, n=40); shift(corr=0.401, p=0.0496, n=40)
- control_time_sampling_pc1_prov: iid(corr=0.433, p=0.0051, n=40); shift(corr=0.433, p=0.0495, n=40)
- control_time_sampling_pc12_prov: iid(corr=0.380, p=0.0151, n=40); shift(corr=0.380, p=0.0241, n=40)
- control_time_loc_coll_occ_prov_macro_area: iid(corr=0.360, p=0.0221, n=40); shift(corr=0.360, p=0.0482, n=40)
- control_time_loc_coll_occ_prov_macro_sections: iid(corr=0.349, p=0.0271, n=40); shift(corr=0.349, p=0.0524, n=40)
- control_time_loc_coll_occ_prov_macro_area_sections: iid(corr=0.241, p=0.139, n=40); shift(corr=0.241, p=0.123, n=40)

## Outputs

- Merged table: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv`
- Stats: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/analysis_results.json`
- Figures: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/figures`

## Notes

- Sampling proxies are derived from PBDB `collection_no` and a coarse environment classifier on PBDB `environment` strings; treat as approximate.
- For final inference, prefer explicit time-series models or block bootstraps, and integrate Macrostrat/rock-area covariates.

