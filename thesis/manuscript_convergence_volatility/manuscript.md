# Draft manuscript: climate volatility and marine functional convergence

## Title (working)

**Deep-time climate volatility predicts marine functional convergence beyond taxonomy across the Phanerozoic**

## Abstract (draft)

Marine provinces can differ taxonomically yet perform similar ecological “jobs.” We test whether deep-time climate volatility acts as a
synchronizing constraint that increases **functional similarity beyond what is expected from shared taxa**. Using Paleobiology Database
(PBDB) occurrences binned at 10 Myr and PBDB ecospace traits (diet, motility, life habit) to define discrete functional roles, we compute
pairwise functional similarity between spatial grid-localities and remove the expected dependence on taxonomic similarity (“functional
excess similarity”). We merge these metrics with an independent Earth-system forcing series derived from CESM paleoclimate snapshots
(Li et al. 2022) and evaluate the relationship between temperature volatility (mean |ΔT| field between successive snapshots) and functional
convergence while controlling for time trends and sampling structure. Across 40 bins spanning ~540–0 Ma, higher volatility predicts higher
marine functional excess similarity. A publication-oriented pair-level regression with bin-clustered robust errors and an exact circular-shift
null (time-series aware) supports a positive volatility “baseline shift”: even taxonomically distinct provinces are more functionally similar
in volatile intervals. Coherence/patchiness metrics correlate with convergence but do not add independent explanatory power beyond volatility
magnitude in this dataset. Simple “low-energy composition shift” mediation is not supported, suggesting convergence is not explained by a
global mean shift in ecospace composition alone. These results support a general macroecological principle: **volatile climates synchronize
functional structure across space**, producing “different taxa, same jobs” at deep-time scales.

## Introduction (draft)

### Functional similarity versus taxonomic similarity in deep time

Two regions can share few taxa yet still be ecologically similar if they contain organisms performing the same broad functions (feeding,
mobility, habitat use). Deep-time work has quantified functional/ecospace occupancy and post-crisis functional restructuring, but a general,
quantitative relationship between *independent physical forcing* and *cross-province functional convergence beyond taxonomy* remains less
well established.

### Volatility as a synchronizing filter

If climate changes are large and rapid (at macro-timescale resolution), they may act as a filter that eliminates or suppresses some strategies
and repeatedly favors robust role mixtures. That could reduce the number of functionally distinct regional solutions, increasing similarity
between distant provinces even when taxa differ.

### Study aim and predictions

We test the prediction:

> **Higher climate volatility increases marine functional similarity beyond taxonomic similarity across provinces.**

We also evaluate mechanistic signatures:
- Whether volatility acts mainly as a **baseline shift** (higher functional similarity even when taxonomy differs) versus changing the
  functional–taxonomic coupling slope.
- Whether simple composition shifts toward “low-energy / sit-and-filter” strategies explain (mediate) the volatility effect.
- Whether spatial coherence of forcing adds explanatory power beyond volatility magnitude.

## Data (draft)

### Fossil occurrences

- Source: PBDB occurrences extracted into a normalized table: `data/processed/merged_occurrences.parquet`.
- Spatial aggregation: 10° × 10° grid localities, using paleocoordinates when available.
- Temporal aggregation: 10 Myr bins (rounded from occurrence midpoints).
- Marine filtering: PBDB ecospace environment string contains “marine”.

### Functional roles (PBDB ecospace)

For each genus, PBDB ecospace assigns:
- `diet`
- `motility`
- `life_habit`

We define a discrete functional role as the tuple `diet|motility|life_habit`.

### Climate forcing

Independent forcing is derived from Li et al. (2022) CESM paleoclimate snapshots (10 Myr spacing):
- Volatility: mean absolute temperature-change magnitude across the global grid between successive snapshots (mean |ΔT| field).

## Methods (draft)

### Taxonomic and functional similarity

Within each time bin:
- For each locality, define the genus set and a role-frequency vector.
- For each pair of localities, compute:
  - Taxonomic similarity: Jaccard similarity of genus sets.
  - Functional similarity: Jensen–Shannon similarity of role-frequency vectors.

### “Functional excess similarity” (convergence beyond taxonomy)

Because functional similarity increases with shared taxa, we estimate the expected relationship and remove it:
- Fit `functional_similarity ~ taxonomic_similarity` across locality pairs.
- Define functional excess similarity as the residual functional similarity relative to the expected value, aggregated per bin.

### Sampling + rock-record controls

To reduce sampling confounding, we include:
- PBDB-derived sampling proxies (localities, marine collections, marine occurrences).
- Macrostrat rock-record proxies (binned section counts and column area).
- Because these are collinear, we compress them into a **sampling PCA index** (PC1/PC2).

### Inference approaches

We report three complementary inferential layers:
1) Bin-level partial correlation tests with time-series-aware circular-shift p-values.
2) Pair-level regression with bin-clustered robust SEs + exact circular-shift null on volatility.
3) Explicit time-series and hierarchical models (bin-level AR errors; pair-level mixed effects by bin).

## Results (draft)

### Main result: volatility predicts functional convergence beyond taxonomy

Primary supported result:
- Higher volatility predicts higher functional excess similarity across marine localities.
  - Pair-level model summary: `thesis/synthesis/output_pair_level_model_volatility_v1/summary.md`
  - Robustness summaries: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb/summary.md`,
    `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/summary.md`

### Mechanistic signature: baseline shift (“different taxa, same jobs”)

Volatility primarily raises the intercept of `functional_similarity ~ taxonomic_similarity` rather than strongly changing its slope:
- `thesis/synthesis/output_function_taxonomy_coupling_v3_binslopes/summary.md`

### Interpretation layer: which ecospace jobs shift?

Under the same sampling+autocorrelation controls, some role mixtures shift with volatility, but many individual-role findings do not survive
multiple-testing correction. A small number of composite tendencies are consistent with increased prevalence of stationary/suspension-feeding
mixtures and reduced prevalence of fast-moving carnivory mixtures:
- `thesis/synthesis/output_role_jobs_volatility_v1/summary.md`

### Mechanism hardening: low-energy composite mediation (negative)

A preregistered “low-energy / sit-and-filter” composite index does not track volatility under circular shifts and does not materially attenuate
the volatility coefficient in the pair-level model:
- `thesis/synthesis/output_low_energy_index_mediation_v1/summary.md`

### Time-series + hierarchical inference

Bin-level OLS/HAC supports `vol_z > 0`, while SARIMAX AR(1) yields a more conservative (non-supported) volatility coefficient; pair-level mixed
effects support `vol_z > 0`:
- `thesis/synthesis/output_time_series_hierarchical_models_v1/summary.md`

## Discussion (draft)

### “So what?”

At 10 Myr resolution across the Phanerozoic, more volatile climates are associated with a higher tendency for far-apart marine provinces to
converge on similar mixes of ecological roles, even when they share few taxa. This supports a deep-time “synchronizing filter” view:
volatility can reduce the space of viable ecological solutions, producing repeated functional structure across regions.

### What it is *not*

The effect is not well explained by a simple global mean shift toward low-energy composition (as defined here), implying that convergence may
be driven more by spatial homogenization/constraint than by changing the overall average ecospace mix.

### Limitations and threats to validity

- PBDB sampling structure and trait completeness remain key limitations despite proxy controls.
- Ecospace categories are coarse and heterogeneous across clades.
- Time-binning (10 Myr) limits resolution and can blur event dynamics.
- AR(1) time-series inference is conservative in this dataset; richer state-space/hierarchical time-series models are a natural next step.

## Reproducibility (draft)

Run all key analyses:

`python thesis/run_all.py`

Key outputs to read:
- `thesis/synthesis/FINAL_REPORT.md`
- `thesis/synthesis/output_pair_level_model_volatility_v1/summary.md`
