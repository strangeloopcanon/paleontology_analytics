# Project roadmap

## Reading order

1. **What did we find?** → [`FINDINGS_SUMMARY.md`](FINDINGS_SUMMARY.md)
2. **Full results with numbers** → [`synthesis/FINAL_REPORT.md`](synthesis/FINAL_REPORT.md)
3. **The paper** → `manuscript_convergence_volatility/manuscript.md` (gitignored; local only)

## How the analysis works

The core question: does climate volatility force taxonomically distinct marine provinces to converge on similar ecological roles?

```
PBDB occurrences (1.97M)          CESM paleoclimate (Li et al. 2022)
        │                                    │
        ▼                                    ▼
 convergence/                        earth_system/climate_540myr/
 run_convergence_analysis.py         → temperature volatility per 10 Myr bin
 → pairwise functional vs                   │
   taxonomic similarity                      │
 → "functional excess similarity"            │
   per 10 Myr bin                            │
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
              synthesis/
              robust_convergence_sampling_autocorr.py
              → merge convergence + volatility + sampling controls
              → circular-shift null, PCA sampling index
              → produces merged.csv (the master table)
                       │
          ┌────────────┼────────────────────┐
          ▼            ▼                    ▼
   robustness_    pair_level_         time_series_
   battery.py     convergence_        hierarchical_
                  model.py            models.py
```

## Folder guide

### Core analysis

| Folder | What it does |
|--------|-------------|
| `convergence/` | Computes pairwise functional and taxonomic similarity across grid localities per time bin. Defines functional excess similarity. Main script: `run_convergence_analysis.py`. |
| `earth_system/climate_540myr/` | Derives the climate forcing time series from CESM NetCDF files. Output: `climate_540myr_timeseries.csv`. |
| `synthesis/` | All downstream inference: merging convergence with forcing, robustness battery, sensitivity tests, pair-level and time-series models. **Start here for results**: `FINAL_REPORT.md`. |
| `manuscript_convergence_volatility/` | Paper draft, supplement, and figure generator. Gitignored — exists only locally until ready. |

### Synthesis scripts (in execution order)

| Script | Purpose |
|--------|---------|
| `robust_convergence_sampling_autocorr.py` | Merges convergence + volatility + sampling controls. Produces `merged.csv`. |
| `pair_level_convergence_model.py` | Pair-level regression with cluster-robust SEs and mixed-effects. |
| `time_series_hierarchical_models.py` | OLS, OLS+HAC, SARIMAX, MixedLM on bin-level data. |
| `ecospace_missingness_diagnostic.py` | Characterises PBDB annotation quality; tests coverage as confound. |
| `robustness_battery.py` | LOO, block bootstrap, Lagerstätten, SARIMAX sweep, HAC, coverage control. |
| `era_heterogeneity_investigation.py` | Tests why the signal concentrates in the Mesozoic. |
| `clade_restriction_test.py` | Reruns convergence for single well-annotated clades. |
| `grid_sensitivity.py` | Tests 10°, 15°, 20° grid resolutions. |
| `terrestrial_convergence_pilot.py` | Pilots convergence in terrestrial vertebrates. |

### Secondary analysis tracks

| Folder | What it does | Status |
|--------|-------------|--------|
| `body_size_stability/` | Dinosaur body-size distributions vs climate volatility. | Exploratory; n=8 bins. |
| `geographic_portfolio/` | Extinction survivorship vs range configuration. | Mixed results; event-dependent. |
| `paleobiotic_velocity/` | Taxon centroid-shift rates with bias controls. | Completed; standalone. |

### Supporting infrastructure

| Folder | What it does |
|--------|-------------|
| `pbdb/` | PBDB download scripts (paginated). Data lands in `data/` (gitignored). |
| `macrostrat/` | Rock-record proxy ingestion and binning. |
| `literature/` | Bibliography, reading lists, cached PDFs. |
| `writeups/` | Early-stage research notes and the initial proposal. |
| `archive/` | Superseded material kept for reference. |

## Data requirements

The repo does not track large datasets. To reproduce, you need:

1. **PBDB occurrences** — run `thesis/pbdb/download_pbdb_occurrences_paged.py` or place pre-built parquets in `data/processed/`.
2. **CESM climate data** — download the Li et al. (2022) NetCDF from [Figshare](https://doi.org/10.6084/m9.figshare.19920662.v1) into `data/raw/external/climate_540myr/`.
3. **Macrostrat** — auto-fetched by `thesis/macrostrat/` scripts.

Once data exists, run `python thesis/run_all.py` to reproduce all results.

## Key outputs (gitignored; generated locally)

| Output | Path |
|--------|------|
| Master merged table | `synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv` |
| Robustness battery | `synthesis/output_robustness_battery/` |
| Ecospace coverage | `synthesis/output_ecospace_missingness/` |
| Era heterogeneity | `synthesis/output_era_heterogeneity/` |
| Clade restriction | `synthesis/output_clade_restriction/` |
| Grid sensitivity | `synthesis/output_grid_sensitivity/` |
| Terrestrial pilot | `synthesis/output_terrestrial_pilot/` |
| Manuscript figures | `manuscript_convergence_volatility/figures/` |
