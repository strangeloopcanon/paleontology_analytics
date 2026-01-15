# Independent forcing test: dinosaur body-size structure vs Earth-system volatility

This merges the Benson et al. (2014) dinosaur body-mass time bins with an independent CESM snapshot series (Li et al. 2022) and tests
whether the “missing middle sizes” metrics covary with climate/paleogeography volatility.

- Body bins: `thesis/body_size_stability/output/body_mass_timebins.csv`
- Earth-system bins: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`
- Merged: `thesis/body_size_stability/output_independent_stability/merged_bodymass_earthsystem.csv`
- Permutations: 10000

## Notes

- Signs: higher `delta_from_prev_*` means *more volatility* between adjacent 10 Myr CESM snapshots.
- If “stability fosters missing-middle bimodality”, we would expect **negative** correlations between volatility and bimodality.

## Files

- Results JSON: `thesis/body_size_stability/output_independent_stability/analysis_results.json`
- Figures: `thesis/body_size_stability/output_independent_stability/figures`

