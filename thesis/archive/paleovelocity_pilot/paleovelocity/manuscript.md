# Deep-time paleovelocity: a mobility proxy from paleocoordinate shifts predicts within-interval genus persistence

**Status:** draft (analysis is reproducible; interpretation + citations need peer review)

## Abstract

Mobility is a plausible buffer against extinction, but deep-time mobility is rarely quantified at scale. Here we introduce *genus paleovelocity* (km/Myr): the great-circle displacement of a genus’ paleogeographic centroid between successive time bins, computed from fossil occurrence paleocoordinates. Using 991,602 unique PBDB occurrences (48,698 genera; 65–535 Ma; 5 Myr bins), we estimate 67,120 capped paleovelocity transitions (99.5th percentile cap at 3,504.6 km/Myr). Median paleovelocity is 418.5 km/Myr (IQR 132.7–1,046.8) with weak correlation to an approximate Phanerozoic temperature curve (Pearson *r*≈0.016 for median-by-bin). In a pooled, genus-held-out logistic model of terminal (right-censored) within-interval extinction, higher paleovelocity is consistently associated with lower terminal extinction odds (odds ratio per 1 SD ≈ 0.86, 95% across-split range 0.85–0.87), alongside broader latitudinal range, environmental breadth, geographic range, and age. These results support paleovelocity as a compact, reproducible mobility proxy that complements abundance/range predictors and can be extended to interval-complete datasets and trait-stratified tests.

## 1. Introduction

Species and clades respond to environmental change through adaptation, tracking, and/or extinction. In modern ecology, *climate velocity* frames the pace at which organisms must move to maintain climate envelopes; an analogous large-scale, deep-time mobility proxy can help test whether more mobile taxa persist longer through crises. Fossil data provide the raw ingredients (taxa, time, space), and PBDB paleocoordinates offer reconstructed positions through time. The goal here is not to reconstruct true dispersal routes, but to build a reproducible, conservative *movement signal* from occurrence geometry and evaluate whether that signal relates to persistence.

## 2. Data and Methods

### 2.1 Data

We use the project’s normalized occurrence table (`data/processed/merged_occurrences.parquet`), primarily derived from PBDB API downloads (default interval setting `Cambrian,Cretaceous`). Data are filtered to rows with non-placeholder genus labels and numeric ages. Overlapping PBDB pulls can duplicate occurrences, so we deduplicate by `(source_db, occurrence_id)` prior to analysis.

After filtering and deduplication, the analysis uses:
- 991,602 unique occurrences
- 48,698 genera
- 95 time bins spanning 535–65 Ma (5 Myr bins; rounded)

### 2.2 Coordinates and time binning

For each occurrence, we define `analysis_lat/analysis_lng` by preferring reconstructed paleocoordinates (`paleolat/paleolng`) and falling back to modern coordinates (`lat/lng`) when paleocoordinates are missing.

Time is discretized into 5 Myr bins using `time_bin = round(mid_ma / 5) * 5`.

### 2.3 Genus centroids and paleovelocity

For each genus × time bin, we compute a paleogeographic centroid:
- latitude: median(`analysis_lat`)
- longitude: circular mean of `analysis_lng` (mean of sin/cos, then `atan2`)

Stepwise paleovelocity is computed between a bin and the previous (older) bin for that genus, restricted to gaps ≤10 Myr:

1) distance = great-circle (haversine) distance between successive centroids (km)  
2) paleovelocity = distance / Δtime (km/Myr)

To reduce sensitivity to extreme centroid shifts, we cap velocities above the 99.5th percentile (3,504.6 km/Myr; 0.5% of transitions).

### 2.4 Within-interval terminal extinction (with right-censoring)

The PBDB query interval truncates the record at ~66 Ma, so a genus whose last appearance falls in the youngest global bin (65 Ma) cannot be classified as extinct within the study window. We treat these genera as *right-censored* and exclude them from extinction modeling.

For remaining genera, we label a genus-bin row as *terminal* if it is the youngest bin in which the genus occurs (within 65–535 Ma).

### 2.5 Statistical model

We fit a pooled logistic regression to predict terminal-bin status using standardized features:
- paleovelocity (km/Myr)
- abundance (occurrence count in bin)
- geographic range (unique 5°×5° localities in bin)
- latitudinal range (max−min latitude in bin)
- environmental breadth (unique environment labels in bin)
- age (number of older bins observed for that genus)

To avoid leakage across repeated observations of the same genus, train/test splits are performed by genus (group shuffle split). We report AUC on held-out genera, repeated across 25 random splits.

## 3. Results

### 3.1 Paleovelocity through time

Paleovelocity varies substantially across bins and taxa (Fig. 1). Across all capped transitions, median paleovelocity is 418.5 km/Myr (IQR 132.7–1,046.8). Median-by-bin paleovelocity shows weak linear correlation with the project’s approximate Phanerozoic temperature curve (*r*≈0.016).

**Figure 1:** `thesis/archive/paleovelocity_pilot/paleovelocity/figures/fig1_velocity_timeseries.png`

### 3.2 Paleovelocity and terminal extinction

Terminal bins exhibit lower paleovelocity than non-terminal bins (median 270.4 vs 470.0 km/Myr; Fig. 2), consistent with mobility being associated with within-interval persistence.

**Figure 2:** `thesis/archive/paleovelocity_pilot/paleovelocity/figures/fig2_terminal_vs_nont_velocity.png`

In the genus-held-out logistic model, predictive performance is stable (AUC mean 0.693; 95% across-split range 0.684–0.700). Feature effect sizes (odds ratio per 1 SD; mean and 2.5–97.5% across splits) indicate consistent protective associations for:
- paleovelocity: OR ≈ 0.86 (0.85–0.87)
- age: OR ≈ 0.61 (0.60–0.62)
- environmental breadth: OR ≈ 0.75 (0.73–0.77)
- latitudinal range: OR ≈ 0.79 (0.77–0.81)
- geographic range: OR ≈ 0.88 (0.85–0.91)

Coefficient summaries (generated on re-run): `thesis/archive/paleovelocity_pilot/output/results/terminal_extinction_logit_coefficients_summary.csv`

### 3.3 Fastest “movers”

A non-inferential ranking of genera by median paleovelocity (minimum 3 transitions) is provided for qualitative follow-up:

`thesis/archive/paleovelocity_pilot/output/tables/top_movers.csv`

## 4. Discussion

This study introduces a lightweight, reproducible mobility proxy derived directly from occurrence geometry and paleocoordinate reconstructions. The near-zero correlation between median paleovelocity and a coarse temperature curve suggests that mobility signals are not trivially explained by this single global proxy, while the consistent negative association between paleovelocity and terminal-bin odds supports the hypothesis that taxa with larger apparent range shifts are less likely to terminate within the observed interval (conditional on abundance/range/breadth).

Key interpretations are intentionally cautious: centroid shifts reflect both real biogeography and changes in sampling/collection geography. The method is best viewed as a *movement signal* suitable for hypothesis generation and for controlled extensions (e.g., clade-specific analyses, interval-complete downloads including the Cenozoic, incorporating sampling proxies, or replacing terminal-bin labels with survivorship models).

## 5. Reproducibility

Run:

```bash
python thesis/archive/paleovelocity_pilot/code/paleovelocity.py --data data/processed/merged_occurrences.parquet --out thesis/archive/paleovelocity_pilot/output
```

Primary outputs:
- Figures: `thesis/archive/paleovelocity_pilot/output/figures/`
- Tables: `thesis/archive/paleovelocity_pilot/output/tables/`
- Results (model + time series): `thesis/archive/paleovelocity_pilot/output/results/`

## 6. Limitations

- **Interval truncation:** this pilot was originally run on a Cambrian–Cretaceous-limited PBDB slice; re-run with an interval-complete local build to reduce boundary censoring.
- **Sampling bias:** centroid shifts mix biological movement with changes in the geographic distribution of sampling.
- **Taxonomic resolution:** genus-level aggregation masks species-level dynamics and lumping/splitting effects.
- **Paleocoordinate uncertainty:** reconstructions depend on PBDB’s rotation model and underlying age/locational uncertainty.
- **Environment labels:** PBDB environment strings are heterogeneous; “breadth” is a coarse proxy.
