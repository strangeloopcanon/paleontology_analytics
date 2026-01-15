# Exploratory results: body-size structure vs biogeographic stability

This is a first-pass test linking a dinosaur body-mass time series (Benson et al. 2014 Dataset S1) to a PBDB-derived
spatial stability proxy (1 - normalized Jensen–Shannon divergence of dinosaur genus-richness grids between adjacent bins).

- Time bin: 10.0 Myr
- Grid: 10.0°
- Permutation test: 10000 shuffles

## Correlation summaries

| Exclude Avialae | Mass variant | n bins | corr(stability,bimodality) | perm-p | corr(stability,gap_ratio) | perm-p |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | mass1 | 9 | 0.464 | 0.214 | -0.882 | 0.0499 |
| 0 | mass2 | 9 | 0.508 | 0.175 | -0.878 | 0.0208 |
| 1 | mass1 | 9 | 0.557 | 0.113 | -0.394 | 0.372 |
| 1 | mass2 | 9 | 0.588 | 0.0945 | -0.364 | 0.426 |

## Files

- Body-mass bins: `thesis/body_size_stability/output/body_mass_timebins.csv`
- Body-mass specimens (all variants): `thesis/body_size_stability/output/body_mass_specimens_all_variants.csv`
- Body-mass specimens (per variant): `thesis/body_size_stability/output/body_mass_specimens_exclAvialae_0_mass1.csv` etc.
- PBDB stability bins: `thesis/body_size_stability/output/pbdb_stability_timebins.csv`
- Merged per-variant bins: `thesis/body_size_stability/output/merged_timebins_exclAvialae_0_mass1.csv`, `thesis/body_size_stability/output/merged_timebins_exclAvialae_0_mass2.csv`, `thesis/body_size_stability/output/merged_timebins_exclAvialae_1_mass1.csv`, `thesis/body_size_stability/output/merged_timebins_exclAvialae_1_mass2.csv`
- Figures: `thesis/body_size_stability/output/figures`

## Interpretation guardrails

- These correlations are **not causal** and may reflect sampling artifacts in either dataset.
- The stability proxy is PBDB-occurrence-based and can move with outcrop/collection focus.
- Treat any signal as a hypothesis generator; next steps should use independent plate/climate stability series and sampling-aware modeling.

