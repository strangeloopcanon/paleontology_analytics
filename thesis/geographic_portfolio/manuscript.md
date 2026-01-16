# Draft manuscript: Geographic portfolio structure and mass‑extinction survivorship

## Working title

**Geographic portfolio structure predicts survivorship across mass extinctions in the fossil record: a paleogeographic connectedness test using PBDB occurrences**

## Abstract (draft)

Geographic range size is a classic buffer against extinction, but range size collapses distinct spatial configurations into a single scalar. Here we test whether **range configuration**—specifically the degree to which a genus’ occupied localities form a single connected “core” versus a fragmented multi‑core distribution—predicts survivorship across major Phanerozoic crises. Using a normalized PBDB occurrence extract (`data/processed/merged_occurrences.parquet`; generated locally / gitignored; 991,602 deduplicated PBDB occurrences after cleaning), we compute genus‑level pre‑event geographic features on paleogeographic grids and model survivorship across four canonical crisis boundaries (~444, 372, 252, 201 Ma). After controlling for sampling‑intensity proxies (occurrence abundance) and geographic range size (grid occupancy), we find that **paleogeographic connectedness is strongly associated with survivorship for the end‑Ordovician and Late Devonian crises at coarse spatial scales**, while **end‑Permian survivorship is better explained by range size and latitudinal/environmental breadth**. Modern‑coordinate negative controls show that some apparent configuration effects can arise from coordinate distortions, underscoring the need to compute geography in paleocoordinates and to report scale sensitivity. These results support a “geographic portfolio” view of extinction vulnerability: when kill mechanisms are geographically heterogeneous, taxa spanning multiple disconnected provinces are more likely to persist, whereas globally uniform crises reduce the benefit of configuration beyond range size.

## 1. Introduction

### 1.1 Background

Deep‑time extinction selectivity research repeatedly implicates **geographic range size** as a survivorship buffer (e.g., `10.1073/pnas.0701257104`). However, two genera with identical range size can differ in configuration: one may occupy a compact connected province, while another spans multiple disconnected provinces or basins. These configurations map naturally to competing mechanisms:

- **Rescue/connectivity** (metapopulation logic): connected distributions can facilitate recolonization and reduce local extinction cascades.
- **Portfolio/spatial heterogeneity**: fragmented multi‑province distributions can buffer against geographically uneven kill mechanisms by reducing synchrony of population loss.

The fossil record is uniquely positioned to test these mechanisms across crisis types, but doing so requires explicit handling of sampling bias and paleogeography.

### 1.2 Research question

> After controlling for range size and sampling intensity, does **pre‑event paleogeographic connectedness** predict which genera survive major crisis boundaries?

We treat connectedness as a *configuration* property (core‑dominated vs multi‑core), not a synonym for range size.

## 2. Data

We use the repo’s normalized occurrence table:

- `data/processed/merged_occurrences.parquet`
- PBDB occurrences only (Neotoma records are excluded by cleaning because genus is missing for those rows).
- Ages are represented as `mid_ma`, binned to 5 Myr by rounding (`time_bin = round(mid_ma/5)*5`).
- Coordinates:
  - **Paleo**: PBDB paleocoordinates where available, otherwise modern coordinates.
  - **Modern (negative control)**: modern coordinates only.

## 3. Methods

### 3.1 Events and “pre” bins

We analyze four canonical crisis boundaries (Ma): 444, 372, 252, 201. For each boundary, we define a “pre” bin as the 5 Myr bin immediately older than the boundary (ceiling to the nearest 5 Myr), and compute pre‑event features from occurrences in that bin.

### 3.2 Geographic features

We compute genus‑level pre‑event features on an equal‑degree grid (5° and a sensitivity at 10°):

- `abundance`: number of occurrences (sampling‑intensity proxy)
- `geographic_range`: number of occupied grid cells (range size proxy)
- `lat_range`: max–min latitude (paleolatitude in paleo mode)
- `env_breadth`: number of unique PBDB environment labels
- `largest_component_frac`: **connectedness** = fraction of occupied grid cells belonging to the largest connected component (4‑neighbor adjacency with longitude wrap)

Connectedness is high for compact core distributions; it declines as distributions become multi‑core / disjunct.

### 3.3 Survivorship outcomes

We compute two outcomes:

- `survived_any`: genus appears in any younger bin after the boundary (range‑through tolerant).
- `survived_10myr`: genus appears within 0–10 Myr after the boundary (stricter, more gap‑sensitive).

### 3.4 Models and controls

For each event, coordinate mode, grid size, and target:

- Fit repeated stratified train/test logistic models (30 splits; 70/30) predicting survivorship.
- Report odds ratios per 1 SD change in each predictor (mean and 95% across splits).
- Baseline model excludes connectedness (`largest_component_frac`) to estimate ΔAUC.

Controls:

- **Modern‑coordinate negative control**: recompute all geographic metrics using modern coordinates to diagnose coordinate/sampling artifacts.
- **Scale sensitivity**: compare 5° vs 10° grids.
- **Clade control**: optional phylum fixed effects (`--with-phylum`) to test whether configuration effects persist within major taxonomic partitions.

Code: `thesis/geographic_portfolio/run_event_portfolio_analysis.py`.

## 4. Results (current)

### 4.1 Range size remains a strong predictor

Across events, `log_geographic_range` generally increases survivorship odds (as expected from prior work), especially for end‑Permian and end‑Triassic crises.

### 4.2 Connectedness (“core‑dominated vs multi‑core”) is event‑ and scale‑dependent

With phylum fixed effects and a **10° grid**, paleocoordinate connectedness shows strong, consistent associations for early events:

- **End‑Ordovician** (444 Ma): connectedness odds ratios < 1 (higher connectedness → lower survivorship), implying that **multi‑core / multi‑province distributions are favored**.
- **Late Devonian** (372 Ma): similar pattern at 10° in paleocoordinates.
- **End‑Permian** (252 Ma): connectedness effects are weak/non‑robust in paleocoordinates at 10°, while latitudinal and environmental breadth remain important.

Modern‑coordinate negative controls attenuate or reverse the connectedness signal for some events at 10° (notably end‑Ordovician and Late Devonian), indicating that configuration inference depends on paleogeographic reconstruction and is not purely a property of modern sampling geography.

Summary figures:

- `thesis/geographic_portfolio/figures/connectedness_or_survived_any.png`
- `thesis/geographic_portfolio/figures/connectedness_or_survived_10myr.png`

### 4.3 Interpretation: different crises “care” about different geographic dimensions

The contrast between (i) strong configuration selectivity at end‑Ordovician / Late Devonian and (ii) weak configuration selectivity at end‑Permian (in paleocoordinates) is consistent with a mechanism difference:

- For crises with strong spatial heterogeneity and/or provinciality, spanning multiple disconnected provinces increases the chance of persistence (portfolio logic).
- For globally uniform crises, configuration adds less beyond range size; breadth metrics may dominate (tolerance/provincial filtering).

### 4.4 Additional hypotheses (tested)

To move beyond the “widespread survives” baseline, we tested five additional, pre‑event geography hypotheses (entropy/evenness, equator crossing, latitudinal position, spatial dispersion, longitudinal span) using the same event-based design with paleocoordinates vs modern-coordinate negative controls and 5° vs 10° grids.

Key takeaways (full results: `thesis/geographic_portfolio/additional_hypotheses_results.md`):

- **Latitudinal position is a major survivorship axis** and can be **sign‑flipped** if you use modern coordinates instead of paleocoordinates (largest effect at end‑Permian).
- **Spatial dispersion shows strong paleo‑specific support at end‑Permian** (coarse grid), but is not a universal “more dispersed is better” rule.
- **Portfolio entropy is strongly positive for end‑Ordovician and (at 10°) Late Devonian**, but becomes negative for end‑Triassic in paleocoordinates.

## 5. Discussion (draft)

### 5.1 So what?

If these results hold under additional controls, the implication is that **“widespread taxa survive” is incomplete**: *how* a taxon is widespread matters, and the relevant configuration depends on the crisis. This reframes geographic buffering as a testable interaction between (i) paleogeographic provinciality and (ii) the spatial footprint of extinction drivers.

### 5.2 Novelty claim (careful framing)

This is presented as a **novel synthesis and scale‑explicit test** rather than a claim that no one has ever discussed geography and extinction. The potentially new contribution is:

- an explicitly defined, reproducible **connectedness/portfolio** metric computed from PBDB paleocoordinates,
- applied *comparably across multiple crisis boundaries* with negative controls and spatial‑scale sensitivity.

### 5.3 Limitations (must be addressed before submission)

- Occurrence‑based survivorship still reflects sampling/preservation; results must be stress‑tested with sampling‑aware methods (e.g., boundary‑crosser variants, gap rules, sampling completeness filters).
- Event boundaries are approximated in 5 Myr bins; tighter stratigraphic alignment (stages) is desirable.
- Phylum fixed effects are crude; stronger trait/life‑mode proxies would sharpen mechanisms.

## 6. Reproducibility

Run (5° default, with phylum):

```bash
python thesis/geographic_portfolio/run_event_portfolio_analysis.py \
  --out thesis/geographic_portfolio/output_with_phylum \
  --with-phylum
```

Grid sensitivity (10°):

```bash
python thesis/geographic_portfolio/run_event_portfolio_analysis.py \
  --out thesis/geographic_portfolio/output_grid10_with_phylum \
  --with-phylum --grid-deg 10
```

Summary figures:

```bash
python thesis/geographic_portfolio/make_summary_figures.py \
  --out thesis/geographic_portfolio/figures
```
