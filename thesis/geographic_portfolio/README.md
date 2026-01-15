# Geographic portfolio & mass‑extinction survivorship (PBDB)

This folder tests a biologically interpretable question that is *adjacent* to (but distinct from) centroid‑velocity “mobility”:

> Do genera with different **geographic portfolio structure** (connected core vs fragmented multi‑core distributions; narrow vs broad latitudinal spread) show different survivorship across major Phanerozoic crises, after controlling for range size and sampling intensity proxies?

The workflow is designed to be reproducible from the repo’s normalized occurrence extract:
`data/processed/merged_occurrences.parquet`.

## What it does

For each crisis boundary (default: ~444, 372, 252, 201 Ma), the pipeline:

1. Selects the 5 Myr time bin immediately **older** than the boundary (“pre” bin).
2. Computes genus‑level pre‑event geographic features on a 5° locality grid:
   - `abundance` (occurrences)
   - `geographic_range` (unique 5° grid cells)
   - `lat_range`
   - `env_breadth`
   - **connectedness** (`largest_component_frac`): fraction of occupied cells in the largest connected component (4‑neighbor adjacency with longitude wrap)
3. Defines survivorship outcomes:
   - `survived_any`: genus appears in *any* younger bin after the boundary (range‑through tolerant; lower false‑extinction rate)
   - `survived_10myr`: genus appears within 0–10 Myr after the boundary (stricter)
4. Fits repeated train/test logistic models (genus rows) to estimate odds ratios and ΔAUC for portfolio features.

## Run

```bash
python thesis/geographic_portfolio/run_event_portfolio_analysis.py \
  --data data/processed/merged_occurrences.parquet \
  --out thesis/geographic_portfolio/output
```

Outputs:
- per‑event datasets: `output/<coords_mode>/<event>_<target>_dataset.parquet`
- model repeats + summaries: `output/<coords_mode>/<event>_<target>_*`
- summary table: `output/summary.csv`

## Additional hypothesis grid

Five additional geography hypotheses (entropy, dispersion, latitude position, longitude span, equator crossing) are tested in:

- `run_additional_hypotheses.py`
- results writeup: `additional_hypotheses_results.md`

Run:

```bash
python thesis/geographic_portfolio/run_additional_hypotheses.py \
  --out thesis/geographic_portfolio/output_additional_hypotheses \
  --with-phylum
```

## Notes / limitations

- “Survivorship” is occurrence‑based; it still reflects sampling/preservation. `survived_any` is less sensitive to short gaps than `survived_10myr`.
- The connectedness metric depends on grid size and adjacency definition; rerun with `--grid-deg` and/or `--time-bin-myr` to test sensitivity.
- Results are intended to be compared for `--coords-mode paleo` vs a modern‑coordinate negative control (`--coords-mode modern`) to diagnose sampling‑geography artifacts.
