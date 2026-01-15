# Research workspace: proposal + self‑critique (v0)

## Working title

**Paleobiotic velocity from PBDB paleocoordinates: separating mobility signals from sampling geography and plate motion, and testing their role in extinction selectivity**

## Core problem

Deep‑time fossil occurrence data encode *where* taxa are found, but also *where rocks are exposed and studied* and how plate motion relocates paleocoordinates through time. Many well‑known extinction predictors (geographic range, environmental breadth, age selectivity) can be estimated from occurrences; however, *mobility* is rarely quantified at PBDB scale in a way that is (a) reproducible, (b) interpretable, and (c) robust to sampling bias.

This project proposes a rigorous, bias‑aware framework for estimating and validating *taxon displacement rates through deep time* and quantifying their association with survivorship.

## Hypotheses (testable)

H1. **Apparent mobility is protective:** higher paleobiotic displacement rate is associated with lower extinction hazard after controlling for abundance, geographic range, environmental breadth, and age.

H2. **Event dependence:** the protective association strengthens during intervals with elevated extinction intensity (e.g., major crises) relative to background.

H3. **Not just sampling geography:** the mobility association persists after controlling for (and/or differencing against) measures of spatiotemporal sampling geography (global sampling centroid shifts, locality density, spatial standardization).

H4. **Coordinate robustness:** results are consistent across (i) paleocoordinates and (ii) modern coordinates used as a negative control, and across centroid definitions (occurrence‑weighted vs locality‑weighted).

## Proposed contributions (deliverables)

### Paper 1 (methods): Mobility metrics + bias controls

- Define a family of mobility estimators from fossil occurrences:
  - occurrence‑weighted centroid displacement rate
  - locality‑weighted centroid displacement rate (reduces oversampling of popular sites)
  - “relative displacement” metrics controlling for global sampling centroid shifts
- Provide uncertainty quantification:
  - bootstrap within genus‑bin (resampling localities)
  - sensitivity to bin width, locality grid size, outlier handling
- Provide null models:
  - within‑bin genus label permutations (preserve sampling geography; destroy biological structure)

### Paper 2 (macroecology): Mobility and extinction hazard

- Fit discrete‑time survival models (genus‑held‑out CV) where the event is disappearance in the next time bin.
- Include time‑varying baseline hazard (time‑bin effects) to avoid confounding by secular sampling and global environmental change.
- Test interactions (mobility × crisis indicators).

### Paper 3 (mechanism): Plate motion vs biotic tracking (optional / stretch)

- Decompose apparent mobility into:
  - plate‑corrected (paleocoordinates) vs present‑day locality coordinates (negative control)
  - residual displacement after removing global sampling centroid movement
- Compare across clades/environments to interpret biological plausibility.

### Paper 3b (biology): Geographic portfolio structure and crisis survivorship

If centroid-velocity “mobility” is not robust to negative controls, a biologically interpretable alternative is to test **range configuration** rather than displacement rate:

- Define a “geographic portfolio / connectedness” metric from occupied paleogeographic grid cells (largest connected component fraction; number of components).
- Test whether configuration predicts boundary-crossing survivorship across major crises **after controlling for range size and sampling intensity proxies**.
- Use modern-coordinate negative controls and spatial-scale sensitivity (e.g., 5° vs 10° grids) as a falsification framework.

Implementation and a draft manuscript live in `geographic_portfolio/`.

## Self‑critique (what could invalidate the project if not solved)

1) **Sampling geography confounding (primary risk):**
   - Centroid shifts can reflect where fossils are collected, not where taxa lived.
   - Mitigation must be explicit and demonstrated (spatial standardization, covariates, null models).

2) **Taxonomic and temporal resolution:**
   - Genus is a coarse unit; PBDB taxonomy and synonymy can induce artificial “movement”.
   - Bin widths trade temporal precision for sampling adequacy; sensitivity analyses must show stability.

3) **Paleocoordinate uncertainty and plate model dependence:**
   - PBDB paleocoordinates depend on rotation models and age estimates; errors propagate into velocities.
   - This work must treat paleobiotic velocity as a *proxy* and quantify robustness.

4) **Extinction definition and censoring:**
   - Interval‑limited downloads can right‑censor survivorship. Models must treat this correctly and ideally use interval‑complete downloads.

5) **Interpretability:**
   - Even if predictive, the mechanism needs careful framing: “mobility proxy” not literal dispersal speed.
   - Negative controls and clade‑stratified validation are required to avoid storytelling.

## What “completion” means for this repo

Within this local `thesis/` workspace, completion means:
- A fully reproducible pipeline that generates figures/tables and model summaries from the local parquet data.
- Robustness + null models showing whether the signal survives plausible confounds.
- A public-facing writeup that clearly separates *results* from *interpretation* and documents limitations.
