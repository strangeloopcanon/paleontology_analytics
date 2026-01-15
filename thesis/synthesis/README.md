# Synthesis: volatility, convergence, and size-structure (working folder)

This folder is a *working* synthesis layer that ties together:

- **Marine functional convergence** measured from PBDB occurrences + PBDB ecospace traits (diet/motility/life habit).
- **Independent climate/paleogeography forcing** from Li et al. (2022) CESM snapshots (derived 10 Myr time series).
- **Dinosaur body-size structure** from Benson et al. (2014) mass estimates, explored against stability/volatility proxies.

Key scripts/results:

- `thesis/synthesis/test_volatility_filter.py` → `thesis/synthesis/output_volatility_filter_v4/summary.md`
- `thesis/convergence/RESULTS.md`
- `thesis/body_size_stability/RESULTS.md`

Notes:
- Everything under `thesis/` is gitignored by design.
- `data/raw/` is also gitignored; large external datasets can be stored there.

