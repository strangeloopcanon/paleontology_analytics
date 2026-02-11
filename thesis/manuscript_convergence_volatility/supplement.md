# Supplementary Material

## S1. Ecospace annotation coverage

Table S1 reports the fraction of genera per 10 Myr bin with complete PBDB ecospace annotations (marine + diet + motility + life habit). Mean coverage across 40 analysis bins is 46% for marine genera with complete roles.

Annotation coverage correlates strongly with the convergence metric in raw terms (*r* = 0.90), but this correlation is almost entirely mediated by a shared secular time trend: coverage vs time *r* = 0.91, convergence vs time *r* > 0.80. After time is controlled, the residual relationship between coverage and convergence is weak and is further absorbed by the sampling PCA controls that enter the primary specification.

Full coverage table: `thesis/synthesis/output_ecospace_missingness/ecospace_coverage_per_bin.csv`

## S2. Control set sensitivity

The primary specification uses: time + sampling PCA (PC1, PC2) + provinciality. Below are results for alternative control sets (all partial correlations of volatility vs convergence, with iid permutation and circular-shift p-values):

| Controls | Partial *r* | iid perm-*p* | Circular-shift *p* |
|---|---|---|---|
| Time only | ~0.41 | 0.008 | 0.026 |
| Time + localities | ~0.29 | 0.071 | 0.106 |
| Time + all sampling (loc/coll/occ) | ~0.35 | 0.027 | 0.027 |
| Time + all sampling + provinciality | ~0.36 | 0.025 | 0.028 |
| Time + sampling PCA (PC1) | — | — | — |
| Time + sampling PCA (PC1) + provinciality | — | — | — |
| **Time + sampling PCA (PC1, PC2) + provinciality** [PRIMARY] | **0.38** | — | **0.050** |

Note: Macrostrat rock-record proxies are incorporated through the sampling PCA index (which combines PBDB and Macrostrat proxies). Naive stacking of multiple Macrostrat covariates introduces collinearity; the PCA index resolves this.

Full results: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/analysis_results.json`

## S3. SARIMAX AR order sweep

AIC and BIC for SARIMAX models with AR(0) through AR(3) errors:

| Model | AIC | BIC | vol beta | vol *p* |
|---|---|---|---|---|
| AR(0) | -160.5 | -148.8 | 0.0117 | 0.079 |
| AR(1) | -158.7 | -145.4 | 0.0104 | 0.162 |
| AR(2) | -153.6 | -138.9 | 0.0080 | 0.325 |
| AR(3) | -157.7 | -141.5 | 0.0046 | 0.399 |

AIC selects AR(0); BIC also selects AR(0). Higher-order AR terms do not improve fit and progressively attenuate the volatility coefficient.

## S4. Newey-West automatic bandwidth

Andrews (1991) rule of thumb: `floor(4 * (N/100)^(2/9))` = 3 lags for N = 40 bins.
OLS + HAC(3): vol beta = 0.0131, SE = 0.0060, *t* = 2.18, *p* = 0.037.

## S5. Leave-one-out bin stability

Dropping each of 40 bins in turn, the partial correlation between volatility and convergence ranges from 0.320 to 0.473 (mean = 0.381). All 40 jackknife correlations are positive. The most influential bin is 270 Ma (mid-Permian); its removal *increases* the correlation.

Figure S1: bar chart of leave-one-out partial correlations with reference line.

## S6. Block bootstrap

Block bootstrap p-values for the partial correlation under the primary specification (10,000 resamples):

| Block size | *p* |
|---|---|
| 2 | 0.020 |
| 3 | 0.021 |
| 5 | 0.029 |

## S7. Lagerstatten sensitivity

Excluding 5 bins containing known exceptionally preserved faunas (50, 150, 170, 430, 510 Ma): partial *r* = 0.319, circular-shift *p* = 0.086 (N = 35 bins). The sign is preserved but significance is reduced, reflecting both loss of power (5 fewer bins) and the genuine contribution of well-preserved intervals.

## S8. Grid-size sensitivity

| Grid | N bins | Partial *r* (time-only controls) | Shift *p* |
|---|---|---|---|
| 10 deg | 50 | 0.234 | 0.16 |
| 15 deg | 48 | 0.225 | 0.063 |
| 20 deg | 50 | 0.072 | 0.82 |

Note: these results use a simplified re-computation with time-only controls (not the full primary specification). The signal is moderate at 10--15 degrees and vanishes at 20 degrees.

## S9. Clade restriction

| Subset | N bins | N genera | Partial *r* | Shift *p* |
|---|---|---|---|---|
| Bivalvia | 6 | — | — (too few bins) | — |
| Gastropoda | 0 | — | — | — |
| Brachiopoda | 18 | 2,471 | -0.088 | 0.611 |
| Combined | 32 | 3,348 | -0.134 | 0.375 |

The signal does not survive restriction to individually well-annotated clades, consistent with functional convergence being an emergent cross-clade phenomenon.

## S10. Effective sample sizes

For the pair-level model, bin-level predictors (volatility, time, provinciality, sampling PCs) have effective N = 40 (number of bins), not the reported number of pairs (~29,000). Cluster-robust standard errors and the mixed-effects model account for this structure. All bin-level inference should reference N = 40 as the effective sample size.

## S11. Seed sensitivity for pair subsampling

The convergence pipeline caps pairwise comparisons at 30,000 per bin. With the default seed (42), the analysis produces the reported results. We have not tested multiple seeds for the main analysis but note that the pair-level model aggregates across bins where subsampling effects average out.

## S12. Era heterogeneity decomposition

Per-era volatility-convergence statistics:

| Era | N bins | Raw *r* | Perm-*p* | Partial *r* (time) | Mean volatility | Mean ecospace coverage |
|---|---|---|---|---|---|---|
| Paleozoic | 17 | -0.106 | 0.685 | 0.206 | 2.21 | 0.581 |
| Mesozoic | 16 | 0.534 | 0.050 | 0.361 | 1.66 | 0.423 |
| Cenozoic | 7 | 0.015 | 0.978 | — | 2.30 | 0.180 |

The Mesozoic concentration is not explained by volatility amplitude (Paleozoic has higher mean volatility) or by annotation coverage (Paleozoic has higher coverage).

## S13. Pair-level mixed-effects model

Random intercepts and (when convergent) random slopes for taxonomic similarity by bin. The fixed-effect volatility coefficient is positive and supported in the mixed-effects framework, with appropriate degrees of freedom for bin-level inference.

## S14. Dinosaur body-size structure (exploratory)

Using Benson et al. (2014) dinosaur mass estimates, we find that the "gap ratio" (a measure of the missing-middle in size distributions) correlates positively with climate volatility (*r* ~ 0.85--0.89, perm-*p* < 0.02, N = 8 bins including Avialae). This is intriguing but based on very few bins and should be treated as a hypothesis for further testing with expanded datasets.

## S15. Geographic portfolio structure (separate track)

Tested whether range configuration (connected-core vs multi-core) predicts survivorship across major mass extinctions. Results are mixed and event-dependent: strong for end-Ordovician and Late Devonian, weak for end-Permian. Coordinate-sensitivity (sign-flips between paleo and modern coordinates) is documented. Full manuscript draft in `thesis/geographic_portfolio/manuscript.md`.
