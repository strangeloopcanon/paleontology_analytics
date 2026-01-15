# Exploratory results: body-size structure vs biogeographic stability

This is a first-pass test linking a dinosaur body-mass time series (Benson et al. 2014 Dataset S1) to a PBDB-derived
spatial stability proxy (1 - normalized Jensen–Shannon divergence of dinosaur genus-richness grids between adjacent bins).

- Time bin: 10.0 Myr
- Grid: 5.0°
- Permutation test: 10000 shuffles

## Correlation summaries

| Exclude Avialae | Mass variant | n bins | corr(stability,bimodality) | perm-p | corr(stability,gap_ratio) | perm-p |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | mass1 | 9 | 0.557 | 0.119 | -0.902 | 0.0263 |
| 0 | mass2 | 9 | 0.587 | 0.0971 | -0.960 | 0.0054 |
| 1 | mass1 | 9 | 0.658 | 0.052 | -0.557 | 0.2 |
| 1 | mass2 | 9 | 0.675 | 0.047 | -0.570 | 0.187 |

## Files

- Body-mass bins: `thesis/body_size_stability/output_bin10_grid5/body_mass_timebins.csv`
- PBDB stability bins: `thesis/body_size_stability/output_bin10_grid5/pbdb_stability_timebins.csv`
- Merged per-variant bins: `thesis/body_size_stability/output_bin10_grid5/merged_timebins_exclAvialae_0_mass1.csv`, `thesis/body_size_stability/output_bin10_grid5/merged_timebins_exclAvialae_0_mass2.csv`, `thesis/body_size_stability/output_bin10_grid5/merged_timebins_exclAvialae_1_mass1.csv`, `thesis/body_size_stability/output_bin10_grid5/merged_timebins_exclAvialae_1_mass2.csv`
- Figures: `thesis/body_size_stability/output_bin10_grid5/figures`

## Interpretation guardrails

- These correlations are **not causal** and may reflect sampling artifacts in either dataset.
- The stability proxy is PBDB-occurrence-based and can move with outcrop/collection focus.
- Treat any signal as a hypothesis generator; next steps should use independent plate/climate stability series and sampling-aware modeling.

