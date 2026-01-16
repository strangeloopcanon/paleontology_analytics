# Independent Earth-system dataset: CESM “540 Myr” snapshots (Li et al. 2022)

This folder stores an **independent** (non‑PBDB) Earth-system dataset we can use as a forcing/stability/volatility series.

Source paper:
- Li, X., Hu, Y., et al. (2022) *A high-resolution climate simulation dataset for the past 540 million years.* Scientific Data.
- Paper URL: `https://www.nature.com/articles/s41597-022-01490-4`
- Figshare record (dataset): `https://doi.org/10.6084/m9.figshare.19920662.v1`

Downloaded artifacts (not tracked):
- `data/raw/external/climate_540myr/High_Resolution_Climate_Simulation_Dataset_540_Myr.nc` (NetCDF; 55 snapshots: 540 Ma → 10 Ma + PI)
- `data/raw/external/climate_540myr/scripts.zip` (author scripts + helper files)
- Extracted upstream scripts (for reference only; not used by our pipeline): `thesis/earth_system/climate_540myr/scripts/`
These are intentionally **not tracked** in git (large upstream artifacts).

## What we derive

`derive_timeseries.py` reads the NetCDF file (via SciPy’s `netcdf_file`) and writes a compact time series:
- Global mean temperature and precipitation (area-weighted)
- Paleogeography summaries from `LANDFRAC` (land fraction), including:
  - global land fraction
  - approximate land “component count” (connected components of `LANDFRAC > 0.5`, with a simple longitude wrap heuristic)
  - coastline index (grid-edge land/sea transitions)
- Volatility metrics as **between-snapshot changes** (e.g., mean absolute ΔLANDFRAC across the globe)

These become independent predictors we can merge onto PBDB-derived time bins (10 Myr) and onto the dinosaur body-size bins (70–200 Ma).

## Run

```bash
python thesis/earth_system/climate_540myr/derive_timeseries.py \
  --nc data/raw/external/climate_540myr/High_Resolution_Climate_Simulation_Dataset_540_Myr.nc \
  --out thesis/earth_system/climate_540myr/output
```

Outputs:
- `output/climate_540myr_timeseries.csv`
- `output/summary.md`
- `output/figures/*.png` (generated; not tracked)
