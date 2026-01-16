# Body-size structure vs “geographic stability” (exploratory)

Goal: a first-pass test of a high-novelty idea:

> Do long periods of geographic/biogeographic stability correlate with a stronger “missing middle sizes” pattern in dinosaur body-size distributions?

This analysis is intentionally exploratory and produces **stored artifacts** (CSV + figures + summary markdown) under `output/`.

Start here: `thesis/body_size_stability/RESULTS.md`

## Data sources

1) **Dinosaur body-mass estimates** from Benson et al. (2014, PLOS Biology) supplementary Dataset S1:
- DOI: `10.1371/journal.pbio.1001853`
- Downloaded from PLOS: `...type=supplementary&id=10.1371/journal.pbio.1001853.s011`
- We use the `Mass estimates` sheet for specimen-level body-mass estimates (`Mass 1`, `Mass 2`) and the `Data.txt` sheet for clade labels and age intervals (stage/epoch names).

2) **PBDB occurrences** from this repo:
- `data/processed/merged_occurrences.parquet`
- Dinosaur proxy: PBDB `class` in `{Saurischia, Ornithischia}` (see caveats).

3) **Time-scale mapping**:
- PBDB interval list API (International Chronostrat scale): `https://paleobiodb.org/data1.2/intervals/list.json?scale=1`
- Used to convert stage/epoch names in Dataset S1 to numeric ages (Ma).

## Key definitions

### “Missing middle sizes” metrics (per time bin)

Computed on specimen-level `log10(body_mass_kg)` within each time bin:
- `bimodality_coeff` (moment-based heuristic)
- `gap_ratio_hist` (histogram valley depth between the two strongest peaks when peaks fall on opposite sides of the median)

### “Geographic stability” metrics (per time bin)

Because we do not (yet) have a clean plate-kinematic “stability” time series, we use PBDB dinosaur occurrences to build a **biogeographic stability proxy**:

- Build a paleogeographic grid (default 10°).
- For each time bin, compute per-cell dinosaur **genus richness** and normalize to a probability distribution over cells.
- Compute **Jensen–Shannon stability** between consecutive bins:
  - JS divergence normalized to `[0,1]`
  - stability = `1 - JS_divergence`

Interpretation: higher values mean the *spatial distribution of dinosaur richness* changes less between adjacent bins.

## Run

```bash
python thesis/body_size_stability/run_analysis.py \
  --out thesis/body_size_stability/output \
  --time-bin-myr 10 \
  --grid-deg 10 \
  --permutations 10000
```

## Outputs

- `output/body_mass_timebins.csv`: size-distribution metrics by time bin.
- `output/pbdb_stability_timebins.csv`: stability metrics by time bin.
- `output/merged_timebins.csv`: merged dataset used for inference.
- `output/summary.md`: results + caveats.
- `output/figures/*.png`: time series + scatterplots.

## Caveats (important)

- The stability proxy is derived from PBDB occurrences and can still reflect sampling and database practice.
- Dataset S1 is a compiled specimen dataset (not PBDB) and has its own sampling biases.
- A strong correlation would be **suggestive**, not causal; next steps would require an independent tectonic/climate stability series and sampling-aware modeling.
