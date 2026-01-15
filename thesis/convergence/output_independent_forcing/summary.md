# Independent forcing test: convergence vs CESM-derived volatility

This merges PBDB ecospace convergence metrics with an independent CESM snapshot series (Li et al. 2022) and tests whether
functional convergence tracks climate and/or paleogeography volatility.

- Convergence bins: `thesis/convergence/output_v2/timebin_metrics.csv`
- Earth-system bins: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`
- Merged: `thesis/convergence/output_independent_forcing/merged_convergence_earthsystem.csv`
- Permutations: 20000

## Results (correlation; permutation p-values)

| Predictor | corr | perm-p | partial corr (| time) | perm-p | n |
|---|---:|---:|---:|---:|---:|
| delta_from_prev_T_global_abs | 0.345 | 0.049 | 0.472 | 0.0057 | 33 |
| delta_from_prev_T_field_meanabs | 0.284 | 0.114 | 0.496 | 0.00345 | 33 |
| delta_from_prev_landfrac_field_meanabs | -0.364 | 0.0384 | 0.041 | 0.818 | 33 |
| delta_from_prev_coastline_abs | -0.304 | 0.0882 | 0.052 | 0.779 | 33 |
| delta_from_prev_land_components_abs | -0.012 | 0.944 | 0.515 | 0.00185 | 33 |

## Figures

- `thesis/convergence/output_independent_forcing/figures`

