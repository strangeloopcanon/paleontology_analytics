# Low-energy / sit-and-filter index (mechanism + mediation)

This defines a *single preregistered composite index* from ecospace categories and tests:
1) whether it tracks volatility (sampling+autocorr-aware), and
2) whether it attenuates the volatility term in the pair-level convergence model.

- bins: 40
- pairs: 27890

## Index definition
- low diet: deposit feeder, detritivore, suspension feeder
- high diet: carnivore, piscivore
- low motility: passively mobile, slow-moving, stationary
- high motility: actively mobile, fast-moving
- low habit: epifaunal, infaunal, semi-infaunal
- high habit: aquatic, aquatic, depth=surface, nektobenthic, nektonic
- index_raw = (diet_low - diet_high) + (mot_low - mot_high) + (hab_low - hab_high); index_z standardized across bins

## Mechanism test (index vs volatility; controls time+sampling PCA+provinciality)
- partial corr = 0.271
- circular-shift p (exact) = 0.200

## Mediation / attenuation (pair-level model)
- vol_z beta (no index) = 0.0185
- vol_z beta (with index) = 0.0175
- attenuation = 0.054
- circular-shift p(vol_z | with index) = 0.025
- circular-shift p(index | with index) = 0.700

## Outputs
- bin index: `thesis/synthesis/output_low_energy_index_mediation_v1/bin_index.csv`
- coefficients (no index): `thesis/synthesis/output_low_energy_index_mediation_v1/coef_no_index.csv`
- coefficients (with index): `thesis/synthesis/output_low_energy_index_mediation_v1/coef_with_index.csv`
- summary JSON: `thesis/synthesis/output_low_energy_index_mediation_v1/summary.json`
- figure: `thesis/synthesis/output_low_energy_index_mediation_v1/figures/index_vs_vol.png`
