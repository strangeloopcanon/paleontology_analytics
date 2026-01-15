# Macrostrat rock-record proxies (sampling covariates)

This folder adds a lightweight Macrostrat-based rock-record proxy time series, intended as a **sampling/rock-record control**
for deep-time PBDB analyses.

## Data source

- Macrostrat API v2 (CC-BY 4.0): `https://macrostrat.org/api/v2/`
- We download gap-bound **sections/packages** from: `https://macrostrat.org/api/v2/sections?all&format=csv`

## Files written (gitignored)

- Raw download: `data/raw/external/macrostrat/sections_all.csv`
- Derived 10 Myr binned time series: `data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv`

## How to reproduce

1) Download the latest sections table:

`python thesis/macrostrat/download_macrostrat_sections.py`

2) Build a binned time series (default: 10 Myr bins; 0–540 Ma):

`python thesis/macrostrat/build_macrostrat_timeseries.py`

## Notes / caveats

- Macrostrat coverage is not globally uniform (North America-heavy); treat as a **sensitivity check**, not a definitive global rock-area series.
- Macrostrat-derived proxies can be highly collinear with PBDB sampling proxies (collections/occurrences/localities); avoid over-interpreting coefficient changes under multicollinearity.

