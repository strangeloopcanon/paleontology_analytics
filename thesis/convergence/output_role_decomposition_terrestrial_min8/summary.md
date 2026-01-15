# Role decomposition: what drives ecospace convergence?

This extends the PBDB ecospace convergence pipeline by decomposing functional similarity into coarse ecospace axes:
`diet`, `motility`, and `life habit` (plus the full role combination).

- Environment filter: `terrestrial` (from PBDB ecospace field `jev`)

Convergence metric: excess similarity = residual of (functional similarity ~ taxonomic similarity) across locality-pairs.

## Key tests vs independent forcing (Li et al. 2022 CESM)

- Volatility series: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv` column `delta_from_prev_T_field_meanabs`
- Convergence decomposition metrics: `thesis/convergence/output_role_decomposition_terrestrial_min8/timebin_metrics_decomposition.csv`

### Correlations (bin-level)

- excess_role_js: corr=-0.037, perm-p=0.963, n=6 ; partial corr=-0.113, perm-p=0.827
- excess_diet_js: corr=0.283, perm-p=0.614, n=6 ; partial corr=0.276, perm-p=0.577
- excess_motility_js: corr=-0.569, perm-p=0.226, n=6 ; partial corr=-0.537, perm-p=0.29
- excess_habit_js: corr=0.203, perm-p=0.743, n=6 ; partial corr=0.090, perm-p=0.876
- entropy_roles: corr=-0.052, perm-p=0.923, n=6 ; partial corr=0.081, perm-p=0.895
