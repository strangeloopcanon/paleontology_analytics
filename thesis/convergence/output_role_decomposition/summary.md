# Role decomposition: what drives ecospace convergence?

This extends the PBDB ecospace convergence pipeline by decomposing functional similarity into coarse ecospace axes:
`diet`, `motility`, and `life habit` (plus the full role combination).

- Environment filter: `marine` (from PBDB ecospace field `jev`)

Convergence metric: excess similarity = residual of (functional similarity ~ taxonomic similarity) across locality-pairs.

## Key tests vs independent forcing (Li et al. 2022 CESM)

- Volatility series: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv` column `delta_from_prev_T_field_meanabs`
- Convergence decomposition metrics: `thesis/convergence/output_role_decomposition/timebin_metrics_decomposition.csv`

### Correlations (bin-level)

- excess_role_js: corr=0.203, perm-p=0.179, n=46 ; partial corr=0.359, perm-p=0.0144
- excess_diet_js: corr=0.271, perm-p=0.0689, n=46 ; partial corr=0.396, perm-p=0.00665
- excess_motility_js: corr=0.251, perm-p=0.0905, n=46 ; partial corr=0.378, perm-p=0.0081
- excess_habit_js: corr=0.282, perm-p=0.0578, n=46 ; partial corr=0.553, perm-p=0.0002
- entropy_roles: corr=0.036, perm-p=0.809, n=46 ; partial corr=0.062, perm-p=0.685

## Which categories get more widespread in volatile climates?

We compare category occupancy across localities between the top and bottom volatility quartiles (by `delta_from_prev_T_field_meanabs`). Deltas are (volatile q75 – stable q25).

### Diet (coarse)
- ↑ suspension feeder: Δ=0.069
- ↑ detritivore: Δ=0.039
- ↑ "photoautotroph": Δ=0.027
- ↑ durophage: Δ=0.009
- ↑ herbivore: Δ=0.006
- ↑ browser, omnivore: Δ=0.004
- ↓ carnivore: Δ=-0.142
- ↓ grazer: Δ=-0.091
- ↓ deposit feeder: Δ=-0.071
- ↓ parasite: Δ=-0.067
- ↓ coprophage: Δ=-0.052
- ↓ piscivore: Δ=-0.027

### Motility (coarse)
- ↑ stationary: Δ=0.104
- ↑ slow-moving: Δ=0.059
- ↑ passively mobile: Δ=-0.003
- ↑ passively mobile, epibiont: Δ=-0.052
- ↑ actively mobile: Δ=-0.075
- ↑ facultatively mobile: Δ=-0.097
- ↓ fast-moving: Δ=-0.211
- ↓ facultatively mobile: Δ=-0.097
- ↓ actively mobile: Δ=-0.075
- ↓ passively mobile, epibiont: Δ=-0.052
- ↓ passively mobile: Δ=-0.003
- ↓ slow-moving: Δ=0.059

### Life habit (coarse)
- ↑ amphibious: Δ=0.035
- ↑ solitary, clonal: Δ=0.031
- ↑ boring: Δ=0.013
- ↑ nektonic: Δ=0.003
- ↑ epifaunal: Δ=0.000
- ↑ planktic: Δ=-0.007
- ↓ nektobenthic, solitary: Δ=-0.246
- ↓ aquatic: Δ=-0.127
- ↓ solitary: Δ=-0.078
- ↓ colonial, clonal: Δ=-0.072
- ↓ colonial: Δ=-0.061
- ↓ aquatic, depth=surface: Δ=-0.055

## Category-by-category (controls time)

For each category, we test whether locality occupancy tracks volatility even after controlling for the strong
long-term time trend (partial correlation; permutation p-values). Only categories present in ≥12 bins are tested.

### Diet (coarse): partial corr(volatility, occupancy | time)
- ↑ suspension feeder: r=0.412, p=0.004, n=46
- ↑ detritivore: r=0.347, p=0.0146, n=46
- ↑ piscivore: r=0.242, p=0.359, n=17
- ↑ "photoautotroph": r=0.206, p=0.17, n=46
- ↑ herbivore: r=0.140, p=0.485, n=26
- ↑ deposit feeder: r=-0.051, p=0.733, n=46
- ↓ coprophage: r=-0.478, p=0.0312, n=20
- ↓ grazer: r=-0.418, p=0.0068, n=39
- ↓ carnivore: r=-0.344, p=0.0194, n=46
- ↓ parasite: r=-0.299, p=0.195, n=21
- ↓ omnivore: r=-0.290, p=0.0572, n=43
- ↓ deposit feeder: r=-0.051, p=0.733, n=46

### Motility (coarse): partial corr(volatility, occupancy | time)
- ↑ stationary: r=0.504, p=0.0006, n=46
- ↑ slow-moving: r=0.363, p=0.0134, n=46
- ↑ actively mobile: r=-0.095, p=0.531, n=46
- ↑ passively mobile: r=-0.226, p=0.143, n=46
- ↑ fast-moving: r=-0.344, p=0.0188, n=46
- ↑ facultatively mobile: r=-0.348, p=0.0184, n=46
- ↓ passively mobile, epibiont: r=-0.469, p=0.036, n=20
- ↓ facultatively mobile: r=-0.348, p=0.0184, n=46
- ↓ fast-moving: r=-0.344, p=0.0188, n=46
- ↓ passively mobile: r=-0.226, p=0.143, n=46
- ↓ actively mobile: r=-0.095, p=0.531, n=46
- ↓ slow-moving: r=0.363, p=0.0134, n=46

### Life habit (coarse): partial corr(volatility, occupancy | time)
- ↑ colonial: r=0.369, p=0.235, n=12
- ↑ epifaunal: r=0.277, p=0.0678, n=46
- ↑ nektobenthic: r=0.098, p=0.512, n=46
- ↑ aquatic, depth=surface: r=-0.029, p=0.907, n=17
- ↑ volant: r=-0.036, p=0.886, n=16
- ↑ aquatic: r=-0.059, p=0.858, n=13
- ↓ solitary: r=-0.350, p=0.144, n=19
- ↓ planktic: r=-0.264, p=0.0776, n=45
- ↓ infaunal: r=-0.172, p=0.258, n=46
- ↓ colonial, clonal: r=-0.141, p=0.561, n=18
- ↓ nektonic: r=-0.088, p=0.555, n=45
- ↓ semi-infaunal: r=-0.076, p=0.676, n=35

## Files

- Pairwise similarities: `thesis/convergence/output_role_decomposition/pairwise_decomposition.csv`
- Time-bin metrics: `thesis/convergence/output_role_decomposition/timebin_metrics_decomposition.csv`
- Category contrasts: `thesis/convergence/output_role_decomposition` (`*_contrast.csv`)
- Figures: `thesis/convergence/output_role_decomposition/figures`

