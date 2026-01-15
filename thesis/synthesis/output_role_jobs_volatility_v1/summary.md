# Role/job drivers under volatility (sampling+autocorr-aware)

This asks: as climate volatility rises, which ecospace categories change in (i) geographic ubiquity and (ii) average within-locality composition?

- bins: 40 (from pair-level model)
- volatility quartiles (mean |ΔT| field): q25=1.232, q75=2.740
- controls: time_z + sampling_pc1_z + sampling_pc2_z + provinciality_z

## Diet (coarse): geographic ubiquity (occupancy fraction)
### Increases with volatility
- parasite: r=0.110, p_shift=0.650, q=0.812, Δq75-q25=0.006
- suspension feeder: r=0.029, p_shift=0.875, q=0.925, Δq75-q25=0.013
- detritivore: r=0.026, p_shift=0.925, q=0.925, Δq75-q25=0.033
### Decreases with volatility
- carnivore: r=-0.431, p_shift=0.050, q=0.250, Δq75-q25=-0.126
- piscivore: r=-0.383, p_shift=0.050, q=0.250, Δq75-q25=-0.022
- grazer: r=-0.297, p_shift=0.150, q=0.438, Δq75-q25=-0.059
- omnivore: r=-0.260, p_shift=0.175, q=0.438, Δq75-q25=0.063
- herbivore: r=-0.236, p_shift=0.350, q=0.700, Δq75-q25=0.002
- photoautotroph: r=-0.140, p_shift=0.450, q=0.750, Δq75-q25=0.021

## Diet (coarse): mean within-locality share
### Increases with volatility
- suspension feeder: r=0.398, p_shift=0.075, q=0.417, Δq75-q25=0.165
- parasite: r=0.288, p_shift=0.250, q=0.458, Δq75-q25=0.004
- detritivore: r=0.205, p_shift=0.250, q=0.458, Δq75-q25=0.019
### Decreases with volatility
- herbivore: r=-0.353, p_shift=0.100, q=0.417, Δq75-q25=-0.004
- deposit feeder: r=-0.309, p_shift=0.125, q=0.417, Δq75-q25=-0.029
- carnivore: r=-0.222, p_shift=0.275, q=0.458, Δq75-q25=-0.132
- omnivore: r=-0.152, p_shift=0.500, q=0.714, Δq75-q25=-0.000
- piscivore: r=-0.092, p_shift=0.650, q=0.812, Δq75-q25=0.001
- grazer: r=-0.062, p_shift=0.750, q=0.833, Δq75-q25=0.000

## Motility (coarse): geographic ubiquity
### Increases with volatility
- slow-moving: r=0.063, p_shift=0.800, q=0.950, Δq75-q25=0.050
- stationary: r=0.042, p_shift=0.950, q=0.950, Δq75-q25=0.029
### Decreases with volatility
- facultatively mobile: r=-0.415, p_shift=0.050, q=0.200, Δq75-q25=-0.111
- actively mobile: r=-0.394, p_shift=0.075, q=0.200, Δq75-q25=-0.081
- passively mobile: r=-0.223, p_shift=0.100, q=0.200, Δq75-q25=-0.021
- fast-moving: r=-0.142, p_shift=0.550, q=0.825, Δq75-q25=-0.250

## Motility (coarse): mean within-locality share
### Increases with volatility
- stationary: r=0.392, p_shift=0.100, q=0.300, Δq75-q25=0.185
- slow-moving: r=0.234, p_shift=0.325, q=0.390, Δq75-q25=0.011
### Decreases with volatility
- facultatively mobile: r=-0.452, p_shift=0.025, q=0.150, Δq75-q25=-0.016
- passively mobile: r=-0.244, p_shift=0.175, q=0.350, Δq75-q25=-0.008
- fast-moving: r=-0.200, p_shift=0.325, q=0.390, Δq75-q25=-0.176
- actively mobile: r=-0.093, p_shift=0.575, q=0.575, Δq75-q25=0.023

## Life habit (coarse): geographic ubiquity
### Increases with volatility
- epifaunal: r=0.027, p_shift=0.925, q=0.925, Δq75-q25=0.006
### Decreases with volatility
- aquatic, depth=surface: r=-0.442, p_shift=0.050, q=0.225, Δq75-q25=0.017
- colonial, clonal: r=-0.342, p_shift=0.075, q=0.225, Δq75-q25=-0.034
- aquatic: r=-0.340, p_shift=0.075, q=0.225, Δq75-q25=0.025
- infaunal: r=-0.331, p_shift=0.075, q=0.225, Δq75-q25=-0.079
- solitary: r=-0.393, p_shift=0.100, q=0.240, Δq75-q25=-0.051
- planktic: r=-0.293, p_shift=0.125, q=0.250, Δq75-q25=0.008

## Life habit (coarse): mean within-locality share
### Increases with volatility
- epifaunal: r=0.311, p_shift=0.100, q=0.400, Δq75-q25=0.035
- nektonic: r=0.072, p_shift=0.725, q=0.845, Δq75-q25=0.021
### Decreases with volatility
- aquatic, depth=surface: r=-0.575, p_shift=0.050, q=0.400, Δq75-q25=-0.003
- aquatic: r=-0.388, p_shift=0.075, q=0.400, Δq75-q25=0.000
- nektobenthic: r=-0.268, p_shift=0.200, q=0.500, Δq75-q25=-0.037
- semi-infaunal: r=-0.276, p_shift=0.225, q=0.500, Δq75-q25=0.008
- planktic: r=-0.234, p_shift=0.250, q=0.500, Δq75-q25=-0.005
- solitary: r=-0.181, p_shift=0.375, q=0.643, Δq75-q25=-0.001

## Full roles: geographic ubiquity
### Increases with volatility
- carnivore, detritivore|slow-moving|epifaunal: r=0.438, p_shift=0.025, q=0.415, Δq75-q25=0.109
- detritivore|slow-moving|epifaunal: r=0.467, p_shift=0.050, q=0.494, Δq75-q25=0.181
- suspension feeder|stationary, attached|low-level epifaunal: r=0.410, p_shift=0.075, q=0.494, Δq75-q25=0.104
- detritivore, grazer|actively mobile|epifaunal: r=0.394, p_shift=0.075, q=0.494, Δq75-q25=0.147
- suspension feeder|stationary|low-level epifaunal: r=0.341, p_shift=0.125, q=0.494, Δq75-q25=0.286
- grazer|facultatively mobile, attached|epifaunal: r=0.303, p_shift=0.125, q=0.494, Δq75-q25=0.046
- suspension feeder|stationary|epifaunal: r=0.272, p_shift=0.125, q=0.494, Δq75-q25=0.042
- suspension feeder, detritivore|actively mobile|shallow infaunal: r=0.243, p_shift=0.125, q=0.494, Δq75-q25=0.036
### Decreases with volatility
- carnivore|fast-moving|low-level epifaunal: r=-0.535, p_shift=0.025, q=0.415, Δq75-q25=-0.405
- suspension feeder|facultatively mobile, attached|low-level epifaunal: r=-0.460, p_shift=0.025, q=0.415, Δq75-q25=-0.035
- grazer, omnivore|slow-moving|low-level epifaunal: r=-0.429, p_shift=0.025, q=0.415, Δq75-q25=-0.011
- photoautotroph|passively mobile|planktonic, depth=surface: r=-0.383, p_shift=0.025, q=0.415, Δq75-q25=0.008
- suspension feeder|facultatively mobile|infaunal: r=-0.438, p_shift=0.050, q=0.494, Δq75-q25=-0.146
- carnivore|actively mobile|aquatic, depth=surface: r=-0.383, p_shift=0.050, q=0.494, Δq75-q25=-0.009
- suspension feeder|stationary, epibiont|epifaunal: r=-0.361, p_shift=0.075, q=0.494, Δq75-q25=-0.040
- photosymbiotic, suspension feeder|stationary, attached|colonial, clonal: r=-0.311, p_shift=0.075, q=0.494, Δq75-q25=-0.035

## Full roles: mean within-locality share
### Increases with volatility
- suspension feeder|actively mobile|epifaunal: r=0.566, p_shift=0.025, q=0.346, Δq75-q25=0.037
- suspension feeder|stationary|low-level epifaunal: r=0.537, p_shift=0.025, q=0.346, Δq75-q25=0.098
- carnivore, detritivore|slow-moving|epifaunal: r=0.473, p_shift=0.025, q=0.346, Δq75-q25=0.010
- suspension feeder|stationary|semi-infaunal: r=0.396, p_shift=0.025, q=0.346, Δq75-q25=0.007
- suspension feeder|stationary, attached|upper-level epifaunal: r=0.440, p_shift=0.050, q=0.415, Δq75-q25=0.012
- suspension feeder|stationary, attached|low-level epifaunal: r=0.429, p_shift=0.050, q=0.415, Δq75-q25=0.019
- suspension feeder|stationary, attached|intermediate-level epifaunal: r=0.409, p_shift=0.050, q=0.415, Δq75-q25=0.013
- detritivore|slow-moving|epifaunal: r=0.407, p_shift=0.100, q=0.607, Δq75-q25=0.017
### Decreases with volatility
- microcarnivore|facultatively mobile|low-level epifaunal: r=-0.474, p_shift=0.025, q=0.346, Δq75-q25=-0.000
- suspension feeder|stationary, attached|colonial, clonal: r=-0.388, p_shift=0.025, q=0.346, Δq75-q25=-0.003
- carnivore|actively mobile|aquatic, depth=surface: r=-0.496, p_shift=0.050, q=0.415, Δq75-q25=-0.007
- carnivore|slow-moving|low-level epifaunal: r=-0.286, p_shift=0.075, q=0.566, Δq75-q25=-0.005
- carnivore|fast-moving|low-level epifaunal: r=-0.375, p_shift=0.125, q=0.607, Δq75-q25=-0.130
- herbivore|stationary|semi-infaunal: r=-0.290, p_shift=0.125, q=0.607, Δq75-q25=-0.001
- microcarnivore|stationary, attached|low-level epifaunal: r=-0.249, p_shift=0.125, q=0.607, Δq75-q25=-0.002
- omnivore|stationary|planktonic: r=-0.310, p_shift=0.150, q=0.607, Δq75-q25=-0.009

## Outputs
- bin controls: `thesis/synthesis/output_role_jobs_volatility_v1/bin_controls.csv`
- locality totals: `thesis/synthesis/output_role_jobs_volatility_v1/bin_localities.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/bin_locality_genera.csv`
- long (occupancy): `thesis/synthesis/output_role_jobs_volatility_v1/diet_occupancy_long.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/motility_occupancy_long.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/habit_occupancy_long.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/role_occupancy_long.csv`
- long (mean locality fractions): `thesis/synthesis/output_role_jobs_volatility_v1/diet_mean_locality_frac_long.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/motility_mean_locality_frac_long.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/habit_mean_locality_frac_long.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/role_mean_locality_frac_long.csv`
- assoc (occupancy): `thesis/synthesis/output_role_jobs_volatility_v1/diet_occupancy_volatility_assoc.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/motility_occupancy_volatility_assoc.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/habit_occupancy_volatility_assoc.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/role_occupancy_volatility_assoc.csv`
- assoc (mean locality fractions): `thesis/synthesis/output_role_jobs_volatility_v1/diet_meanfrac_volatility_assoc.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/motility_meanfrac_volatility_assoc.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/habit_meanfrac_volatility_assoc.csv`, `thesis/synthesis/output_role_jobs_volatility_v1/role_meanfrac_volatility_assoc.csv`
