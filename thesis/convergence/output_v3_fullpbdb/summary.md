# Exploratory results: functional convergence using PBDB ecospace

This run uses PBDB occurrences (from the repo parquet) combined with PBDB taxon ecospace annotations (diet/motility/life habit/environment).

- Time bin: 10.0 Myr
- Grid: 10.0°
- Genus inclusion: ≥ 5 PBDB occurrences in the repo dataset
- Locality inclusion: ≥ 25 unique genera per locality×bin (after de-duplication)

## Hypotheses (bin-level tests, permutation p-values)

- H1 (post-perturbation proxy; JS residual): corr(turnover_from_prev, functional_excess_similarity_js) = 0.104, perm-p = 0.522, n = 40
- H1 (post-perturbation proxy; role Jaccard residual): corr(turnover_from_prev, functional_excess_similarity_roles_jaccard) = 0.143, perm-p = 0.377, n = 40
- H2 (fragmentation/provinciality; JS residual): corr(provinciality, functional_excess_similarity_js) = -0.098, perm-p = 0.546, n = 40
- H2 (fragmentation/provinciality; role Jaccard residual): corr(provinciality, functional_excess_similarity_roles_jaccard) = -0.181, perm-p = 0.263, n = 40
- Trend: corr(time_bin, functional_excess_similarity_js) = 0.926, perm-p = 5e-05
- Trend: corr(time_bin, functional_excess_similarity_roles_jaccard) = 0.946, perm-p = 5e-05
- Partial (controls time): corr(provinciality, convergence_js | time) = 0.324, perm-p = 0.0392
- Partial (controls time): corr(turnover_from_prev, convergence_js | time) = -0.328, perm-p = 0.0398
- H3 (volatility): currently uses the same turnover proxy as H1; will be re-tested with an independent climate/paleogeography volatility series.

## Files

- Ecospace mapping: `thesis/convergence/output_v3_fullpbdb/ecospace_genus_mapping.csv`
- Bin metrics: `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`
- Pair sample: `thesis/convergence/output_v3_fullpbdb/pairwise_sample.csv`
- Figures: `thesis/convergence/output_v3_fullpbdb/figures`

## Interpretation guardrails

- Ecospace annotations have missingness and may not be uniformly curated across clades/time.
- PBDB occurrences reflect sampling/rock availability; treat this as a hypothesis generator unless sampling and independent forcing are incorporated.

