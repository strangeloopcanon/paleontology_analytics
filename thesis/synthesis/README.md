# Synthesis: volatility, convergence, and size-structure (working folder)

This folder is a *working* synthesis layer that ties together:

- **Marine functional convergence** measured from PBDB occurrences + PBDB ecospace traits (diet/motility/life habit).
- **Independent climate/paleogeography forcing** from Li et al. (2022) CESM snapshots (derived 10 Myr time series).
- **Dinosaur body-size structure** from Benson et al. (2014) mass estimates, explored against stability/volatility proxies.

Key scripts/results:

- Start here: `thesis/synthesis/FINAL_REPORT.md`
- `thesis/synthesis/test_volatility_filter.py` → `thesis/synthesis/output_volatility_filter_v4/summary.md`
- `thesis/convergence/RESULTS.md`
- `thesis/body_size_stability/RESULTS.md`

Notes:
- Under `thesis/`, we track writeups + code + a small set of curated figures.
- Large datasets and per-run exports live under `data/` and `output*` folders and are not tracked by default.
