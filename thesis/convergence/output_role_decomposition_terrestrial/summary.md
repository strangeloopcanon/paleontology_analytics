# Role decomposition: what drives ecospace convergence?

This extends the PBDB ecospace convergence pipeline by decomposing functional similarity into coarse ecospace axes:
`diet`, `motility`, and `life habit` (plus the full role combination).

- Environment filter: `terrestrial` (from PBDB ecospace field `jev`)

Convergence metric: excess similarity = residual of (functional similarity ~ taxonomic similarity) across locality-pairs.

## Key tests vs independent forcing (Li et al. 2022 CESM)

- Volatility series: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv` column `delta_from_prev_T_field_meanabs`
- Convergence decomposition metrics: `thesis/convergence/output_role_decomposition_terrestrial/timebin_metrics_decomposition.csv`

### Correlations (bin-level)

- excess_role_js: corr=nan, perm-p=nan, n=2 ; partial corr=nan, perm-p=nan
- excess_diet_js: corr=nan, perm-p=nan, n=2 ; partial corr=nan, perm-p=nan
- excess_motility_js: corr=nan, perm-p=nan, n=2 ; partial corr=nan, perm-p=nan
- excess_habit_js: corr=nan, perm-p=nan, n=2 ; partial corr=nan, perm-p=nan
- entropy_roles: corr=nan, perm-p=nan, n=2 ; partial corr=nan, perm-p=nan
