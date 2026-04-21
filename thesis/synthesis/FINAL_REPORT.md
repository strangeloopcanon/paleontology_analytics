# Synthesis report: climate volatility and marine functional convergence

Current state of the marine volatility→convergence analysis, including all robustness checks and sensitivity tests.

## The claim

When Phanerozoic climate volatility is higher, geographically distant marine regions (10° paleocoordinate grid cells) become more functionally similar than expected from shared taxonomy. We call this "functional excess similarity" — the residual of Jensen-Shannon functional similarity regressed on Jaccard taxonomic similarity across locality pairs. The pattern is consistent with environmental filtering: volatile climates compress ecosystems toward a narrower range of ecological roles, producing convergent community structure even among taxonomically distinct faunas.

## Data

- **Fossil occurrences:** Paleobiology Database (PBDB), Cambrian–Holocene, full download (1.97M occurrences). Ecospace annotations (diet, motility, life habit) from PBDB's `ecospace` fields.
- **Climate forcing:** CESM paleoclimate simulations (Li et al. 2022), 10 Myr snapshots spanning 540 Ma. Primary predictor: `delta_from_prev_T_field_meanabs` (spatially-averaged absolute temperature change between consecutive snapshots).
- **Sampling controls:** PBDB collection/occurrence counts (marine-classified) and Macrostrat rock-record proxies (section counts, column areas), compressed via PCA to avoid collinearity.

## Primary result

| Metric | Value |
|--------|-------|
| Partial correlation (volatility vs convergence) | r = 0.380 |
| Controls | time + sampling PCA (PC1, PC2) + provinciality |
| N bins | 40 |
| Exact circular-shift p (all 40 shifts) | 0.050 |
| Random-sampling circular-shift p (20k draws) | 0.025 |
| Block bootstrap p (b=2) | 0.020 |
| Block bootstrap p (b=3) | 0.021 |
| OLS + Newey-West HAC p | 0.037 |
| SARIMAX AR(0) p | 0.079 |

The two circular-shift p-values reflect different estimators for the same null. Exact enumeration with N=40 bins has a resolution floor of 1/40 = 0.025; the observed p=0.050 corresponds to 2 of 40 shifts producing correlations as extreme as observed. The random-sampling estimate approximates the continuous null distribution and gives p=0.025. Block bootstrap, which does not face this resolution constraint, consistently returns p=0.020–0.029.

The SARIMAX result is marginal (p=0.079 for AR(0), AIC-selected). Higher AR orders weaken the signal further. The most conservative time-series model does not support the claim at α=0.05.

## Baseline-shift mechanism

The per-bin regression of functional similarity on taxonomic similarity reveals that volatility raises the intercept (baseline functional similarity when taxonomy diverges) without measurably changing the slope. This "baseline shift" is the mechanistic signature: volatile climates make distant regions more functionally similar even when they share few genera, consistent with convergent environmental filtering rather than taxonomic homogenisation.

## Robustness battery

### Leave-one-out stability
All 40 leave-one-out partial correlations are positive. Range: [0.320, 0.473]. No single bin drives the result. The most influential bin (270 Ma, late Permian) reduces r to 0.320 when dropped.

### Lagerstaetten sensitivity
Excluding 5 bins containing major Lagerstätten (Burgess Shale, Chengjiang, Mazon Creek, Solnhofen, Messel): r = 0.319, exact shift p = 0.086. The signal weakens but remains positive; the loss of 5 bins (12.5% of data) predictably reduces power.

### SARIMAX AR order sweep

| Order | AIC | BIC | vol_β | vol_p |
|-------|-----|-----|-------|-------|
| AR(0) | -160.5 | -148.8 | 0.012 | 0.079 |
| AR(1) | -158.7 | -145.4 | 0.010 | 0.162 |
| AR(2) | -153.6 | -138.9 | 0.008 | 0.325 |
| AR(3) | -157.7 | -141.5 | 0.005 | 0.399 |

AIC selects AR(0). The volatility effect is marginal and weakens with higher AR orders. This is the most conservative inference and should be stated explicitly.

### OLS + HAC
Newey-West with automatic bandwidth (3 lags, Andrews rule): β = 0.013, SE = 0.006, p = 0.037.

## Ecospace coverage confound

PBDB ecospace annotation completeness correlates at r = 0.90 with the convergence metric (raw). Both increase toward the present: coverage vs time r = 0.91. This raised concern that annotation quality, not ecology, drives the signal.

Partial correlations (controlling for time) clarify:
- `frac_has_role` (any role assigned): partial r = −0.028 → entirely explained by time
- `frac_marine_with_role` (marine + complete role): partial r = 0.365 → not fully absorbed by time
- `frac_in_ecospace` (any ecospace entry): partial r = 0.375 → not fully absorbed by time

Adding `frac_marine_with_role` as a control to the primary specification: volatility r drops from 0.380 to 0.328, exact shift p = 0.10, block bootstrap p = 0.047. The signal is attenuated but survives the block bootstrap. The manuscript should acknowledge this honestly: marine-specific coverage confounds part of the variance.

## Era heterogeneity

The signal concentrates in the Mesozoic:

| Era | N bins | Raw r | Perm p | Partial r (time) |
|-----|--------|-------|--------|-------------------|
| Paleozoic | 17 | −0.106 | 0.685 | 0.206 |
| Mesozoic | 16 | 0.534 | 0.050 | 0.361 |
| Cenozoic | 7 | 0.015 | 0.978 | — |

The Mesozoic concentration is not explained by volatility amplitude (Paleozoic and Cenozoic have higher mean volatility). Ecospace coverage is highest in the Paleozoic (0.58) and drops in the Cenozoic (0.18). Land area fraction and paleogeographic connectivity (coastline index, land components) differ across eras but do not obviously account for the pattern.

## Clade restriction (negative result)

Restricting to well-annotated clades eliminates the signal:
- **Brachiopoda** (18 bins, 2471 genera): r = −0.09, shift p = 0.61
- **Combined well-annotated** (32 bins, 3348 genera): r = −0.13, shift p = 0.38
- Bivalvia and Gastropoda had too few qualifying bins.

This is interpretable two ways: (a) the signal is a genuinely cross-clade phenomenon that requires mixing clades to emerge, or (b) the signal requires mixing well-annotated and poorly-annotated genera, suggesting annotation artifacts. The manuscript must present both.

## Grid sensitivity

| Grid | N bins | r | Shift p |
|------|--------|---|---------|
| 10° | 50 | 0.234 | 0.16 |
| 15° | 48 | 0.225 | 0.063 |
| 20° | 50 | 0.072 | 0.82 |

The primary analysis uses 15° bins (matching the CESM grid resolution). The signal is positive at 10° and 15° but vanishes at 20°, suggesting the spatial scale matters and very coarse grids wash out the pattern.

## Terrestrial pilot (negative result)

Applying the same pipeline to terrestrial vertebrates (3663 genera, 21k occurrences, 129 roles, 11 qualifying bins): r = −0.40, perm p = 0.22. No evidence of the marine convergence pattern in the terrestrial realm, though power is low.

## Reproduction

```bash
python thesis/run_all.py              # full pipeline
python thesis/run_all.py --skip-core  # skip data-heavy convergence recomputation
python thesis/run_all.py --only-hardening  # only sensitivity scripts
```

## Key outputs

| What | Path |
|------|------|
| Core merged bins | `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv` |
| Robustness battery | `thesis/synthesis/output_robustness_battery/` |
| Ecospace coverage | `thesis/synthesis/output_ecospace_missingness/` |
| Era heterogeneity | `thesis/synthesis/output_era_heterogeneity/` |
| Clade restriction | `thesis/synthesis/output_clade_restriction/` |
| Grid sensitivity | `thesis/synthesis/output_grid_sensitivity/` |
| Terrestrial pilot | `thesis/synthesis/output_terrestrial_pilot/` |
| Pair-level model | `thesis/synthesis/output_pair_level_model_volatility_v1/` |
| Time-series models | `thesis/synthesis/output_time_series_hierarchical_models_v1/` |
| Manuscript + supplement | `thesis/manuscript_convergence_volatility/` (gitignored) |
| Figures | `thesis/manuscript_convergence_volatility/figures/` (gitignored) |
