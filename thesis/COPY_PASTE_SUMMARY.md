# Copy/paste summary of work done (repo: `paleontology_analytics`)

This is a compact, model-friendly summary of what has been built and what we found so far. All artifacts live under `thesis/`
(gitignored). No commits/pushes were made.

## Data available in-repo

- PBDB occurrences (full 0–540 Ma span matching the climate series; includes Cenozoic slice that was missing initially):
  - Canonical: `data/processed/pbdb_occurrences.parquet`
  - Extended (keeps sampling-critical fields like `collection_no`): `data/processed/pbdb_occurrences_extended.parquet`
  - Merged analysis table: `data/processed/merged_occurrences.parquet`
- Raw PBDB CSV snapshots (ignored): `data/raw/`

## External datasets added (ignored)

- Li et al. (2022) CESM “past 540 Myr” climate snapshot NetCDF:
  - Stored: `data/raw/external/climate_540myr/High_Resolution_Climate_Simulation_Dataset_540_Myr.nc`
  - Paper: `10.1038/s41597-022-01490-4`
  - Dataset DOI: `10.6084/m9.figshare.19920662.v1`
- Macrostrat rock-record proxies (used as sampling/rock-record sensitivity control):
  - Processed 10 Myr-binned series: `data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv`
- Dinosaur body mass dataset (cached under thesis outputs):
  - Benson et al. (2014) Dataset S1 (used for dinosaur mass): `10.1371/journal.pbio.1001853`

## Major analysis tracks + results

### 1) Mass extinction survivorship vs “geographic portfolio”

Folder: `thesis/geographic_portfolio/`

Lay hypothesis: taxa with more “geographic portfolio structure” (multi-core / fragmented occupancy) are more likely to survive
mass-extinction boundaries.

Status/result: mixed/weak and not very stable across events; consistent with huge exogenous forcing + sampling confounds.
Writeups:
- `thesis/geographic_portfolio/manuscript.md`
- `thesis/geographic_portfolio/additional_hypotheses_results.md`

### 2) Dinosaur “missing-middle” body-size structure vs stability/volatility

Folder: `thesis/body_size_stability/`

What we computed:
- Per 10 Myr bin dinosaur body-size distribution metrics (using Benson 2014 masses), including a “gap” metric:
  - `gap_ratio_hist` (higher = weaker missing-middle / more filled-in middle)
  - `bimodality_coeff`

PBDB-derived “stability” proxy (first pass; not independent):
- Suggestive only at small bins; disappears with coarser binning (10 vs 20 Myr) → consistent with timescale sensitivity.
Key outputs:
- `thesis/body_size_stability/output/summary.md`
- `thesis/body_size_stability/RESULTS.md`

Independent forcing test (important):
- Merge dinosaur 10 Myr bins with independent CESM volatility series; test correlations.
- Strongest signal: **more climate volatility → weaker missing-middle** (Avialae included; mass2):
  - `corr(delta_from_prev_T_field_meanabs, gap_ratio_hist) ≈ +0.853`, perm‑p ≈ `0.009`, `n=8`
Outputs:
- `thesis/body_size_stability/output_independent_stability/summary.md`
- `thesis/body_size_stability/output_independent_stability/analysis_results.json`
- `thesis/body_size_stability/output_independent_stability/merged_bodymass_earthsystem.csv`

Literature anchor already downloaded:
- O’Gorman & Hone 2012 (dinosaur size distribution/modality): `10.1371/journal.pone.0051925`
  - PDF cached: `thesis/literature/pdfs/10.1371_journal.pone.0051925.pdf`

### 3) Functional convergence across provinces (PBDB ecospace)

Folder: `thesis/convergence/`

Core idea:
- Define functional “roles” per genus from PBDB ecospace (diet/motility/life habit/environment).
- Compute region-pair functional similarity vs taxonomic similarity; define **functional excess similarity** as residual:
  - `functional_excess_similarity_js` (JS residual)

Main supported “insight” result:
- Using full PBDB (40×10 Myr bins, marine), higher climate volatility predicts higher **functional excess similarity**
  (i.e., distant provinces become *more functionally similar than expected from taxonomy*).
- Robustness stack:
  - PBDB sampling proxies (localities/collections/occurrences) + provinciality controls
  - Circular-shift null to respect time-series autocorrelation
  - Macrostrat rock/area proxies handled via a **sampling PCA index** (to avoid collinearity blowups)
Key outputs:
- Core convergence bins: `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`
- Robustness summary (PBDB sampling + circular shift): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb/summary.md`
- Robustness (+ Macrostrat, PCA sampling index): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/summary.md`
- Publication-oriented pair-level model (cluster-robust + exact circular-shift p-values): `thesis/synthesis/output_pair_level_model_volatility_v1/summary.md`

Role decomposition (what dimensions drive it?):
- Decomposed similarity into coarse `diet`, `motility`, `life habit` axes.
- Earlier (time-only) decomposition suggested life habit was strongest; later publication stack centers on the pair-level model + PCA sampling.
Outputs:
- `thesis/convergence/output_role_decomposition/summary.md`
- `thesis/convergence/output_role_decomposition/timebin_metrics_decomposition.csv`
- `thesis/convergence/output_role_decomposition/*occupancy_timeseries.csv`

Mechanistic signatures that held up best:
- **Baseline shift / “different taxa, same jobs”**: volatility raises the intercept of
  `functional_similarity ~ taxonomic_similarity` (higher functional similarity even when taxonomy differs), rather than clearly changing the slope.
  - `thesis/synthesis/output_function_taxonomy_coupling_v3_binslopes/summary.md`
- Simple “roles become less clade-specific” probe via MI/NMI did **not** support increased interchangeability:
  - `thesis/synthesis/output_role_interchangeability_mi_v1/summary.md`
- Interpretability (“which jobs change”) under the *same* sampling+autocorr controls (exploratory; many tests):
  - `thesis/synthesis/output_role_jobs_volatility_v1/summary.md`
- Pre-registered “low-energy / sit-and-filter” composite index (mechanism attempt):
  - Tested index vs volatility (controls + circular shifts) and mediation in the pair-level model.
  - Result: index is **not supported** under circular shifts and does not materially attenuate the volatility term (≈5%).
  - `thesis/synthesis/output_low_energy_index_mediation_v1/summary.md`

### 4) Coherence-beats-magnitude (explored; not independently supported)

Idea: spatial coherence of climate change might matter more than magnitude.

Result:
- Several coherence/patchiness metrics correlate with convergence, but in this dataset they are strongly correlated with magnitude,
  and do not add independent explanatory power once magnitude is included.
- `thesis/synthesis/output_coherence_beats_volatility/summary.md`

## Literature review artifacts

- Auto-built broad OpenAlex dump: `thesis/literature/reading_lists/reading_list_openalex.md`
- Heuristic curated shortlist: `thesis/literature/shortlist.md`
- Working novelty memo: `thesis/synthesis/NOVELTY_REVIEW.md`

## High-level “so what” (current best articulation)

- Marine ecosystems: when climate swings more between 10 Myr snapshots, distant provinces become **more functionally similar than
  taxonomy would predict** (“different species, same jobs”). Best-supported signature is a **baseline shift**: even taxonomically distinct
  provinces converge on similar mixes of ecospace roles.
- Dinosaurs: in more volatile climates, dinosaur body-size structure shows **less of a missing-middle**; stability appears to allow
  more persistent niche partitioning/extremes (barbell-like structure) at ~10 Myr scales.

## What’s needed next to make this publication-grade

1) Expand from residualization/circular-shifts to explicit time-series or hierarchical models (AR errors / state-space; bin random effects).
2) (Optional) Stronger terrestrial functional dataset to test generality beyond marine (current PBDB terrestrial ecospace is thin).
3) Mechanism hardening: pre-register and test a small number of biologically motivated “low-energy vs high-energy” composite indices rather
   than many per-role tests.

## New inference hardening (done)

- Explicit time-series + hierarchical model fits (bin-level OLS/HAC/SARIMAX AR(1); pair-level MixedLM by bin):
  - `thesis/synthesis/output_time_series_hierarchical_models_v1/summary.md`
