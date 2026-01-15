# Derived time series: climate + paleogeography (Li et al. 2022 CESM snapshots)

- NetCDF: `data/raw/external/climate_540myr/High_Resolution_Climate_Simulation_Dataset_540_Myr.nc`
- Output CSV: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`

## Notes

- Simulations are ordered as in the authors’ `extract_data.ncl`: 540 Ma, 530 Ma, ..., 10 Ma, PI (0 Ma).
- `LANDFRAC` is used as a paleogeography proxy (land–sea distribution).
- Volatility metrics are “from previous snapshot” differences (10 Myr step).

## Variables

| Column | Meaning |
|---|---|
| `T_global_mean_c` | area-weighted global mean monthly-mean surface temperature |
| `P_global_mean_mm_month` | area-weighted global mean monthly-mean precipitation |
| `land_area_fraction` | area-weighted mean land fraction |
| `land_components` | approximate number of land components (`LANDFRAC > threshold`) |
| `coastline_index` | grid-edge land/sea transition count (proxy for coastline complexity) |
| `delta_from_prev_*` | absolute change from previous (older) snapshot |
| `delta_from_prev_T_coherence_ratio` | coherence proxy: `|Δ global mean T| / mean(|ΔT field|)` (≈1 means mostly same-sign change globally) |
| `delta_from_prev_T_sign_agreement_frac` | coherence proxy: fraction of cells whose ΔT sign matches the global mean ΔT sign |
| `delta_from_prev_T_sign_edge_count` | patchiness proxy: number of adjacent grid edges where ΔT sign flips |
| `delta_from_prev_T_sign_components` | patchiness proxy: number of connected components in warming + cooling sign masks |
| `delta_from_prev_T_morans_i` | patchiness proxy: Moran’s I of ΔT field (4-neighbor) |
| `delta_from_prev_T_pc1_frac` | coherence proxy: rank-1 dominance of ΔT field (SVD pc1 energy fraction) |
| `delta_from_prev_T_effective_rank` | coherence proxy: effective rank of ΔT field (lower = more low-dimensional) |
| `delta_from_prev_T_participation_ratio` | coherence proxy: participation ratio of ΔT field singular spectrum |

## Figures

- `thesis/earth_system/climate_540myr/output/figures`

