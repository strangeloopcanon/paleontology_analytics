# Draft research writeup (local, gitignored): Paleobiotic velocity and extinction selectivity

## Status

This is a *local* draft writeup designed to be iterated. It is **not** yet publication-ready; it documents what was done, what the data support, and what remains to reach a defensible, public-facing result.

## Abstract (draft)

Mobility is a plausible buffer against extinction, but deep-time mobility is difficult to quantify from biased fossil occurrence records. Building on climate-change velocity and biotic velocity frameworks (e.g., Loarie et al. 2009 `10.1038/nature08649`; Ordóñez & Williams 2013 `10.1111/ele.12110`), this work develops and stress-tests an occurrence-derived *paleobiotic velocity* proxy: the great-circle displacement rate of taxon paleogeographic centroids through time derived from PBDB paleocoordinates. Using a normalized PBDB occurrence extract (991,602 unique occurrences; 48,698 genera; 5 Myr bins; 535–65 Ma), a pilot analysis suggested a protective association between higher paleovelocity and within-interval terminal extinction odds. A more rigorous discrete-time survival framework with genus-held-out evaluation, time-bin fixed effects, sampling-geometry covariates, coordinate negative controls, and centroid-weighting sensitivity tests shows that (i) mobility coefficients remain directionally protective (odds ratio < 1 per 1 SD), but (ii) incremental predictive gain over established covariates (range, breadth, age) is small and similar under modern-coordinate negative controls. These results imply that naïve mobility signals can be substantially confounded by sampling geography, motivating stronger spatial standardization and null model tests as necessary conditions for causal interpretation.

## 1. Background and motivation

### 1.1 Climate velocity → biotic velocity

Climate velocity formalizes how quickly climates move across space (Loarie et al. 2009 `10.1038/nature08649`; Hamann et al. 2014 `10.1111/gcb.12736`) and has been linked to biodiversity patterns and observed range shifts (Sandel et al. 2011 `10.1126/science.1210173`; Pinsky et al. 2013 `10.1126/science.1239352`; García Molinos et al. 2015 `10.1038/nclimate2769`), while also showing systematic pitfalls (Dobrowski & Parks 2016 `10.1038/ncomms12349`; Chivers et al. 2017 `10.1038/ncomms14434`).

Biotic velocity extends the concept to the pace of community/taxon distribution change itself (Ordóñez & Williams 2013 `10.1111/ele.12110`; Carroll et al. 2015 `10.1371/journal.pone.0140486`).

### 1.2 Why deep-time “mobility” is hard

Fossil occurrences reflect both organismal history *and* sampling: outcrop area, research effort, preservation, and database practice. Current calls for spatial standardization emphasize that spatiotemporal sampling can drive macroecological patterns if not controlled (Antell et al. 2024 `10.1017/pab.2023.36`). Methods for estimating turnover from fossil occurrences often explicitly model sampling (Connolly & Miller 2001 `10.1666/0094-8373(2001)027<0751:JEOSAT>2.0.CO;2`; Silvestro et al. 2014 `10.1093/sysbio/syu006`; PyRate `10.1111/2041-210X.12263`).

### 1.3 What would count as “signal”

A defensible deep-time mobility proxy should:
1) be reproducible from raw data and explicit assumptions,
2) show robustness across reasonable binning/weighting choices,
3) behave differently than sampling-only negative controls,
4) persist when controlling for known predictors (range/breadth/age selectivity; e.g., Payne & Finnegan 2007 `10.1073/pnas.0701257104`; Heim & Peters 2011 `10.1371/journal.pone.0018946`; Finnegan et al. 2008 `10.1666/07008.1`).

## 2. Data and resources

### 2.1 Data source

This repo uses a normalized occurrence extract (`data/processed/merged_occurrences.parquet`) with columns including genus, age, environment, modern coordinates, and PBDB paleocoordinates. PBDB programmatic access is documented in Peters & McClennen 2015 (`10.1017/pab.2015.39`), with a broader user guide (Uhen et al. 2023 `10.5070/p9401160531`).

### 2.2 Study window

The current dataset spans approximately **535–65 Ma** at 5 Myr resolution (rounded). It is interval-limited (Cambrian–Cretaceous) and does not provide complete post-66 Ma follow-up, which affects censoring at the youngest bin.

## 3. Methods

### 3.1 Pilot mobility metric (archived)

The initial implementation (“paleovelocity pilot”) computed genus × bin paleogeographic centroids and stepwise velocities between successive bins.

Pilot artifacts are preserved in:
- `pilot/paleovelocity/` (writeup + figures + tables)
- `pilot/code/paleovelocity.py` (original script)

### 3.2 Rigorous pipeline (current)

The rigorous pipeline (this work) adds:
- multiple centroid weightings (occurrence vs locality)
- explicit global sampling centroid estimates per bin
- alignment of genus displacement with global sampling displacement
- discrete-time survival modeling with genus-held-out evaluation
- modern-coordinate “negative control” mobility metric

Pipeline code:
- `paleobiotic_velocity/run_pipeline.py`
- `paleobiotic_velocity/posthoc_models.py` (additional model variants)
- `paleobiotic_velocity/event_interactions.py` (crisis-window interaction checks)

Run instructions are in `paleobiotic_velocity/README.md`.

### 3.3 Extinction modeling approach

Rather than terminal-bin classification, we model *discrete-time extinction hazard*:

For each genus present in time bin *t*, define the next younger global bin *t+1*. The outcome is 1 if the genus is absent in *t+1*, else 0. Rows at the youngest global bin are excluded because the next interval is unobserved (right-censoring at the dataset boundary).

Models use group-wise splits by genus to avoid leakage.

## 4. Results (current)

### 4.1 Pilot results (for context)

Pilot summary (see `pilot/paleovelocity/results/terminal_extinction_logit_metrics.json`):
- centroid-shift mobility distributions were broad (median ~418 km/Myr after capping)
- terminal vs non-terminal bins differed in mobility
- terminal extinction classifier AUC was ~0.69

### 4.2 Rigorous survival models: what changes

Key result: **mobility remains directionally protective (OR < 1), but adds minimal incremental predictive value** once time-bin fixed effects and established covariates are included.

Example scenario summary (see `paleobiotic_velocity/output/summary.csv`):
- AUC(full) ~0.797–0.798 across scenarios with time fixed effects
- AUC(baseline without mobility) ~0.795–0.797
- ΔAUC attributable to mobility is small (~0.001–0.0015)

Restricted to genera with ≥3 bins (posthoc; see `paleobiotic_velocity/output/paleo_locality_5myr_5deg/results/posthoc/posthoc_summary.csv`):
- With time fixed effects, AUC ~0.793; ΔAUC ~0.00055
- Without time fixed effects, AUC ~0.701; ΔAUC ~0.00043

### 4.3 Negative control

A critical finding is that the **modern-coordinate mobility negative control shows similar mobility odds ratios and ΔAUC** to the paleocoordinate version (see `paleobiotic_velocity/output/modern_occurrence_5myr_5deg_negative_control/`), implying substantial confounding by sampling geography and/or methodological structure.

### 4.4 Crisis interactions (exploratory)

Coarse crisis-window interaction tests (±10 Myr around 444, 372, 252, 201 Ma) did **not** show a stable mobility×crisis interaction effect (see `.../results/event_interactions/*_coef_summary.csv`).

### 4.5 Null model: centroid permutation within time bins

To test whether the mobility effect could be produced by the marginal spatial sampling distribution within each time bin, we ran a permutation null that **shuffles genus centroids within each time bin**, then recomputes velocities and refits the hazard model (time fixed effects; same covariates).

For `paleobiotic_velocity/output/paleo_locality_5myr_5deg`:
- observed mean mobility odds ratio (10 splits): ~0.88  
- centroid-permutation null (10 permutations × 10 splits): mean mobility odds ratios ~0.95–0.98  
- none of 10 permutations produced an odds ratio as protective as observed (empirical *p*≈0 with this small permutation count)

This suggests the protective mobility association is not explained solely by the per-bin centroid distribution; it depends on genus-level temporal structure. However, the same pattern also appears under the modern-coordinate negative control, so additional controls are required before attributing the signal to paleogeography-specific processes.

### 4.6 New direction: geographic portfolio structure across crisis boundaries

Because the mobility signal is difficult to interpret causally given negative-control similarity, a parallel (and more biologically interpretable) line of work tests **range configuration** rather than centroid velocity:

> Do genera with the same range size but different *configuration* (compact connected core vs fragmented multi-core / multi-province distributions) show different survivorship across major crises?

This work lives in `geographic_portfolio/` and is event-based (end‑Ordovician, Late Devonian, end‑Permian, end‑Triassic), using paleocoordinate connectedness (`largest_component_frac`) with modern-coordinate negative controls and spatial-scale sensitivity (5° vs 10°).

Early results suggest the connectedness signal is **event- and scale-dependent**, with the strongest paleocoordinate-specific configuration effects emerging for the end‑Ordovician and Late Devonian at coarser (10°) grids, while end‑Permian survivorship is more strongly associated with range size and latitudinal/environmental breadth than with connectedness.

## 5. Interpretation and critique

1) The pilot “terminal extinction” formulation can exaggerate interpretability because it does not encode time-varying hazard and can be sensitive to interval truncation.

2) In the rigorous hazard model, mobility is not a dominant predictor once range/breadth/age and time effects are included. This is consistent with the idea that mobility is partly redundant with range-based measures and sampling structure.

3) The similarity between paleocoordinate and modern-coordinate mobility effects is a warning sign: a substantial part of the “mobility” signal may be driven by where fossils are found rather than where taxa moved in life.

## 6. What remains for a publication-ready result

To make this work defensible at a publication-review standard, the next steps must directly attack sampling confounding:

- Implement explicit spatial standardization per time bin (e.g., equal-area grids; rarefaction across paleoregions) and re-estimate mobility.
- Add null models that preserve sampling geography while destroying biological structure (within-bin genus label permutations at the occurrence level).
- Extend to interval-complete downloads that include the Cenozoic to address censoring and test generality.
- Add external validation: clade-focused mobility proxies (traits, larval dispersal mode) where available.
- Consider sampling-aware extinction estimators (CMR/PyRate-inspired) to decouple disappearance from preservation.

## Appendix: bibliography

The bibliography build is in `literature/`:
- `literature/core_dois.txt`
- `literature/references.bib`
- `literature/bibliography.md`
