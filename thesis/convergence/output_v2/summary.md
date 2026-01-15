# Exploratory results: functional convergence using PBDB ecospace

This run uses PBDB occurrences (from the repo parquet) combined with PBDB taxon ecospace annotations (diet/motility/life habit/environment).

- Time bin: 10.0 Myr
- Grid: 10.0°
- Genus inclusion: ≥ 5 PBDB occurrences in the repo dataset
- Locality inclusion: ≥ 25 unique genera per locality×bin (after de-duplication)

## Hypotheses (bin-level tests, permutation p-values)

- H1 (post-perturbation proxy; JS residual): corr(turnover_from_prev, functional_excess_similarity_js) = 0.042, perm-p = 0.81, n = 33
- H1 (post-perturbation proxy; role Jaccard residual): corr(turnover_from_prev, functional_excess_similarity_roles_jaccard) = 0.094, perm-p = 0.603, n = 33
- H2 (fragmentation/provinciality; JS residual): corr(provinciality, functional_excess_similarity_js) = 0.070, perm-p = 0.692, n = 33
- H2 (fragmentation/provinciality; role Jaccard residual): corr(provinciality, functional_excess_similarity_roles_jaccard) = -0.055, perm-p = 0.762, n = 33
- Trend: corr(time_bin, functional_excess_similarity_js) = 0.890, perm-p = 5e-05
- Trend: corr(time_bin, functional_excess_similarity_roles_jaccard) = 0.928, perm-p = 5e-05
- Partial (controls time): corr(provinciality, convergence_js | time) = 0.388, perm-p = 0.0245
- Partial (controls time): corr(turnover_from_prev, convergence_js | time) = -0.305, perm-p = 0.0833
- H3 (volatility): currently uses the same turnover proxy as H1; will be re-tested with an independent climate/paleogeography volatility series.

## Files

- Ecospace mapping: `thesis/convergence/output_v2/ecospace_genus_mapping.csv`
- Bin metrics: `thesis/convergence/output_v2/timebin_metrics.csv`
- Pair sample: `thesis/convergence/output_v2/pairwise_sample.csv`
- Figures: `thesis/convergence/output_v2/figures`

## Interpretation guardrails

- Ecospace annotations have missingness and may not be uniformly curated across clades/time.
- PBDB occurrences reflect sampling/rock availability; treat this as a hypothesis generator unless sampling and independent forcing are incorporated.

