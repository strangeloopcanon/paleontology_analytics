# Five additional hypotheses (beyond connectedness)

These hypotheses are designed to be testable with the current repo dataset (`data/processed/merged_occurrences.parquet`) using the same event-based design as `run_event_portfolio_analysis.py`.

All are evaluated **at the pre-event bin** for each crisis (default: ~444, 372, 252, 201 Ma), controlling for:

- sampling-intensity proxy: `log_abundance`
- range size proxy: `log_geographic_range`
- latitudinal spread: `lat_range`
- environment-label breadth: `log_env_breadth`
- portfolio connectedness (prior result): `largest_component_frac`
- optional clade control: `phylum` fixed effects (`--with-phylum`)

## H1. Portfolio evenness (entropy) hypothesis

**Claim:** for a fixed range size, genera whose occupied cells are distributed more evenly across multiple disconnected components have higher survivorship (a “portfolio” buffer against spatially heterogeneous killing).

**Operationalization:** `component_entropy` (Shannon entropy of component size shares on the pre-event grid).

## H2. Equator-crossing hypothesis

**Claim:** genera spanning both hemispheres pre-event survive more often because they occupy multiple climate belts/refugia.

**Operationalization:** `cross_equator` = 1 if `lat_min < 0 < lat_max` in the pre-event bin, else 0.

## H3. Latitudinal position (refugia) hypothesis

**Claim:** where a genus is centered latitudinally matters (e.g., high-lat refugia or tropical buffering), beyond how wide its range is.

**Operationalization:** `centroid_abs_lat` = `abs(centroid_lat)` in the pre-event bin.

## H4. Spatial dispersion hypothesis

**Claim:** more spatially dispersed ranges (not just larger) have higher survivorship by reducing geographic synchrony of losses.

**Operationalization:** `log_dispersion_km` = `log1p(mean distance from occupied grid cells to the genus centroid)`.

## H5. Longitudinal span hypothesis

**Claim:** broader longitudinal coverage (spanning basins/provinces) increases survivorship in crises with strong geographic structure.

**Operationalization:** `lon_span_deg` = minimal circular longitude span (0–360°) covering all occupied grid-cell longitudes.

## How these are tested

Implemented in `thesis/geographic_portfolio/run_additional_hypotheses.py` and evaluated under:

- paleocoordinates vs modern-coordinate negative controls
- 5° vs 10° grid sensitivity
- two survivorship targets: `survived_any`, `survived_10myr`

