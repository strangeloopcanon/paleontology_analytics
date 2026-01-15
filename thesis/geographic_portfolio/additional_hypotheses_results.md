# Results: five additional hypotheses (tested)

All tests use `run_additional_hypotheses.py` with:

- events: end‑Ordovician (444 Ma), Late Devonian (372 Ma), end‑Permian (252 Ma), end‑Triassic (201 Ma)
- pre‑event features on 5° and 10° grids
- paleocoordinates vs modern-coordinate negative controls
- outcome: `survived_any` and `survived_10myr`
- model: repeated stratified logit (30 splits), **with phylum fixed effects**
- baseline covariates: `log_abundance`, `log_geographic_range`, `lat_range`, `log_env_breadth`, `largest_component_frac`

Full summary table: `output_additional_hypotheses/summary.csv`.

## H1 Portfolio entropy (component entropy)

**Supported for early crises; direction flips for end‑Triassic.**

- **End‑Ordovician:** strong positive association in paleo (e.g., `survived_any`, 5°: OR≈2.25 [1.78, 2.98]), also present but smaller in modern negative controls at 10° (OR≈1.67 [1.36, 2.12]).
- **Late Devonian:** positive in paleo at 10° (e.g., `survived_any`: OR≈1.85 [1.29, 2.55]) and *not* significant in modern controls.
- **End‑Triassic:** higher entropy is associated with *lower* survivorship in paleo (OR<1 across grids/targets; modern mostly null).
- **End‑Permian:** not supported (mostly null in both coordinate modes).

Interpretation: a “portfolio/evenness” advantage appears in some crises (especially early ones) but is not universal.

## H2 Equator crossing

**Not supported as a general buffer.**

- Effects are inconsistent across events and often appear (negatively) in modern controls; the only “paleo‑only” signal is a weak negative association for Late Devonian `survived_any`.

Interpretation: simply spanning both hemispheres is not a stable predictor of crisis survivorship in this dataset/design.

## H3 Latitudinal position (absolute latitude of the pre‑event centroid)

**Strong, crisis‑dependent signal, and critically coordinate‑dependent (often sign‑flipping).**

- **End‑Permian:** very strong in paleo, with large predictive contribution (e.g., `survived_any`, 5°: OR≈0.64 [0.60, 0.67], ΔAUC≈+0.015), while modern coordinates give the opposite sign (OR≈1.22 [1.14, 1.30]).
- **End‑Ordovician:** paleo suggests lower‑|lat| (more equatorial) is protective (OR<1), while modern suggests the opposite (OR>1).
- **End‑Triassic:** positive in both modes (higher |lat| → higher survivorship).

Interpretation: you can get **qualitatively wrong latitudinal selectivity** if you do not use paleogeographic reconstruction; the end‑Permian result is especially large.

## H4 Spatial dispersion (mean distance of occupied cells from the centroid)

**Event‑dependent; strongest paleo‑specific support at end‑Permian.**

- **End‑Permian:** strong positive association in paleo at 10° (e.g., `survived_any`: OR≈1.64 [1.46, 1.86], ΔAUC≈+0.007) and not significant in modern controls.
- **Late Devonian:** positive in both paleo and modern (modern stronger), suggesting confounding/sampling geometry for that event at this scale.
- **End‑Ordovician:** negative in paleo at 10° (OR<1), modern null.

Interpretation: dispersion looks like a real survivorship axis for end‑Permian (in paleocoordinates), but is not a universal “more dispersed is better” rule.

## H5 Longitudinal span (circular longitude coverage)

**Mixed; often significant but adds little predictive value and changes sign across events.**

- **End‑Ordovician:** positive in both paleo and modern (OR>1), suggesting this may be partly geometric/sampling-driven.
- **End‑Permian:** negative in paleo (OR<1) while modern is mostly null.
- ΔAUC is frequently ~0 or negative even when coefficients are significant.

Interpretation: longitude span is better treated as a descriptive axis than a robust survivorship predictor in the current model.

## Bottom line

Across the five new hypotheses, the ones that “hold up” best under negative controls and scale checks are:

1) **Latitudinal position** (very strong, but interpretation requires paleocoordinates; signs can flip in modern controls).  
2) **Spatial dispersion** for the **end‑Permian** (paleo‑specific at coarse scale).  
3) **Portfolio entropy** for **end‑Ordovician / Late Devonian** (with some leakage into modern at end‑Ordovician).

