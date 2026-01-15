# Exploratory results: body-size structure vs biogeographic stability

This is a first-pass test linking a dinosaur body-mass time series (Benson et al. 2014 Dataset S1) to a PBDB-derived
spatial stability proxy (1 - normalized Jensen–Shannon divergence of dinosaur genus-richness grids between adjacent bins).

- Time bin: 20.0 Myr
- Grid: 10.0°
- Permutation test: 10000 shuffles

## Correlation summaries

| Exclude Avialae | Mass variant | n bins | corr(stability,bimodality) | perm-p | corr(stability,gap_ratio) | perm-p |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | mass1 | 7 | -0.061 | 0.891 | 0.166 | 0.868 |
| 0 | mass2 | 6 | 0.155 | 0.722 | -0.007 | 1 |
| 1 | mass1 | 7 | -0.066 | 0.879 | 0.467 | 0.502 |
| 1 | mass2 | 6 | 0.149 | 0.738 | 0.400 | 0.531 |

## Files

- Body-mass bins: `thesis/body_size_stability/output_bin20_grid10/body_mass_timebins.csv`
- PBDB stability bins: `thesis/body_size_stability/output_bin20_grid10/pbdb_stability_timebins.csv`
- Merged per-variant bins: `thesis/body_size_stability/output_bin20_grid10/merged_timebins_exclAvialae_0_mass1.csv`, `thesis/body_size_stability/output_bin20_grid10/merged_timebins_exclAvialae_0_mass2.csv`, `thesis/body_size_stability/output_bin20_grid10/merged_timebins_exclAvialae_1_mass1.csv`, `thesis/body_size_stability/output_bin20_grid10/merged_timebins_exclAvialae_1_mass2.csv`
- Figures: `thesis/body_size_stability/output_bin20_grid10/figures`

## Interpretation guardrails

- These correlations are **not causal** and may reflect sampling artifacts in either dataset.
- The stability proxy is PBDB-occurrence-based and can move with outcrop/collection focus.
- Treat any signal as a hypothesis generator; next steps should use independent plate/climate stability series and sampling-aware modeling.

