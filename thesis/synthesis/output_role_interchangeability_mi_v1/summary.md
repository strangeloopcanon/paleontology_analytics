# Role interchangeability under volatility (MI/NMI; first pass)

Hypothesis: climate volatility increases taxon↔role interchangeability (roles become less clade-specific).

We compute per-bin taxon↔role association strength using mutual information (MI) between `family` (or `order`) and `role_id`.
Interchangeability index = `1 - NMI` (higher = roles are less taxon-specific).

Spatial scope: same bins + localities as the main marine convergence analysis:
- bins: `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`
- locality allowlist built from: `thesis/convergence/output_v3_fullpbdb/pairwise_sample.csv`

Inputs:
- PBDB occurrences: `data/processed/pbdb_occurrences_extended.parquet`
- PBDB ecospace mapping: `thesis/convergence/output_v3_fullpbdb/ecospace_genus_mapping.csv`
- Earth-system forcing: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`
- Macrostrat proxies: `data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv`

Taxon↔role metric used for the main test:
- `1 - nmi_family_role_genus_sqrt` (genus-presence weighting; families vs roles).

Sampling control:
- sampling PCA PC1 explained variance: 0.695

## Partial correlation tests (volatility vs interchangeability)

- IID permutation p-values shuffle residuals (exchangeable bins).
- Circular-shift p-values preserve autocorrelation structure (time-ordered bins).

- control_time: iid(corr=0.087, p=0.594, n=40); shift(corr=0.087, p=0.623, n=40)
- control_time_pc1: iid(corr=0.073, p=0.662, n=40); shift(corr=0.073, p=0.695, n=40)
- control_time_pc12: iid(corr=0.047, p=0.778, n=40); shift(corr=0.047, p=0.949, n=40)
- control_time_pc12_prov: iid(corr=0.044, p=0.787, n=40); shift(corr=0.044, p=0.95, n=40)

## Outputs

- MI time bins: `thesis/synthesis/output_role_interchangeability_mi_v1/timebin_role_interchangeability.csv`
- Merged table: `thesis/synthesis/output_role_interchangeability_mi_v1/merged.csv`
- Stats: `thesis/synthesis/output_role_interchangeability_mi_v1/analysis_results.json`
- Sampling PCA: `thesis/synthesis/output_role_interchangeability_mi_v1/sampling_pca.json`
- Figures: `thesis/synthesis/output_role_interchangeability_mi_v1/figures`

## Notes

- This is a bin-level analysis; publication-grade inference should move to a pair-level / hierarchical model and a sampling-aware MI estimator if needed.
- `NMI` is sensitive to the taxonomic level chosen; `order`-level metrics are included in `timebin_role_interchangeability.csv` for comparison.

