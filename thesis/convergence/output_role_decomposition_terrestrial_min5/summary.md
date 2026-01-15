# Role decomposition: what drives ecospace convergence?

This extends the PBDB ecospace convergence pipeline by decomposing functional similarity into coarse ecospace axes:
`diet`, `motility`, and `life habit` (plus the full role combination).

- Environment filter: `terrestrial` (from PBDB ecospace field `jev`)

Convergence metric: excess similarity = residual of (functional similarity ~ taxonomic similarity) across locality-pairs.

## Key tests vs independent forcing (Li et al. 2022 CESM)

- Volatility series: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv` column `delta_from_prev_T_field_meanabs`
- Convergence decomposition metrics: `thesis/convergence/output_role_decomposition_terrestrial_min5/timebin_metrics_decomposition.csv`

### Correlations (bin-level)

- excess_role_js: corr=0.049, perm-p=0.884, n=11 ; partial corr=0.053, perm-p=0.875
- excess_diet_js: corr=-0.185, perm-p=0.59, n=11 ; partial corr=-0.185, perm-p=0.579
- excess_motility_js: corr=0.019, perm-p=0.955, n=11 ; partial corr=-0.003, perm-p=0.992
- excess_habit_js: corr=-0.044, perm-p=0.898, n=11 ; partial corr=-0.036, perm-p=0.917
- entropy_roles: corr=-0.026, perm-p=0.939, n=11 ; partial corr=-0.053, perm-p=0.876

## Which categories get more widespread in volatile climates?

We compare category occupancy across localities between the top and bottom volatility quartiles (by `delta_from_prev_T_field_meanabs`). Deltas are (volatile q75 – stable q25).

### Diet (coarse)
- ↑ herbivore: Δ=0.192
- ↑ piscivore: Δ=0.115
- ↑ carnivore: Δ=0.024
- ↑ insectivore: Δ=0.016
- ↑ durophage: Δ=-0.050
- ↑ omnivore, frugivore: Δ=-0.050
- ↓ grazer: Δ=-0.219
- ↓ omnivore: Δ=-0.217
- ↓ detritivore: Δ=-0.213
- ↓ durophage: Δ=-0.050
- ↓ omnivore, frugivore: Δ=-0.050
- ↓ insectivore: Δ=0.016

### Motility (coarse)
- ↑ fast-moving: Δ=0.017
- ↑ actively mobile: Δ=-0.008
- ↑ facultatively mobile: Δ=-0.219
- ↓ facultatively mobile: Δ=-0.219
- ↓ actively mobile: Δ=-0.008
- ↓ fast-moving: Δ=0.017

### Life habit (coarse)
- ↑ ground dwelling, solitary: Δ=0.207
- ↑ arboreal, solitary: Δ=0.205
- ↑ ground dwelling, gregarious: Δ=0.154
- ↑ ground dwelling: Δ=0.114
- ↑ scansorial: Δ=-0.009
- ↑ amphibious: Δ=-0.022
- ↓ aquatic: Δ=-0.480
- ↓ epifaunal: Δ=-0.219
- ↓ volant, solitary: Δ=-0.213
- ↓ ground dwelling, depth=surface: Δ=-0.117
- ↓ volant: Δ=-0.073
- ↓ arboreal: Δ=-0.072

## Files

- Pairwise similarities: `thesis/convergence/output_role_decomposition_terrestrial_min5/pairwise_decomposition.csv`
- Time-bin metrics: `thesis/convergence/output_role_decomposition_terrestrial_min5/timebin_metrics_decomposition.csv`
- Category contrasts: `thesis/convergence/output_role_decomposition_terrestrial_min5` (`*_contrast.csv`)
- Figures: `thesis/convergence/output_role_decomposition_terrestrial_min5/figures`

