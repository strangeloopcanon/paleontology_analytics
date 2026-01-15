# Terrestrial extensions (PBDB occs ecospace) — current status

## Why earlier terrestrial runs were underpowered

Our original “terrestrial” attempt used **genus-level** PBDB `taxa/list` ecospace (`show=ecospace`) and required complete
`jdt+jmo+jlh`. Terrestrial genera have heavy missingness in `jdt` and `jlh`, so only ~600 genera were usable, which yielded too few
time bins.

PBDB actually provides **occurrence-level** ecospace fields via `occs/list` when `show=ecospace` is requested, and those are much
more complete for vertebrates (e.g., Mammalia).

## Datasets downloaded (raw; gitignored)

- Mammalia occurrences with ecospace:
  - `data/raw/pbdb_occurrences_mammalia_ecospace_paged.csv`
  - Source: PBDB `occs/list.csv` with `base_name=Mammalia` and `show=...ecospace`
- Tetrapoda occurrences with ecospace:
  - `data/raw/pbdb_occurrences_tetrapoda_ecospace_paged.csv`
  - Source: PBDB `occs/list.csv` with `base_name=Tetrapoda` and `show=...ecospace`

Downloader used:
- `thesis/pbdb/download_pbdb_occurrences_paged.py`

## Convergence pipeline (occ-level ecospace)

New runner:
- `thesis/convergence/run_convergence_analysis_occ_ecospace.py`

It mirrors the marine pipeline but uses occ-level ecospace columns:
- `diet`, `motility`, `life_habit` (role = `diet|motility|life_habit`)
- taxonomic similarity = Jaccard on genus sets
- functional similarity = Jensen–Shannon similarity on role-frequency vectors
- “functional excess similarity” = residual from global fit `functional_similarity ~ taxonomic_similarity`, averaged by bin

## Results so far

### Terrestrial tetrapods (first pass)

- Convergence bins: `thesis/convergence/output_occ_ecospace_tetrapoda_terr_min5_pairs50/timebin_metrics.csv`
- Robustness summary (sampling proxies from same PBDB export + circular shifts):
  - `thesis/synthesis/output_convergence_sampling_autocorr_tetrapoda_terr_v1/summary.md`

Outcome: positive correlations (≈0.2–0.38 depending on controls) but **not statistically supported** at n=17 bins once sampling +
provinciality controls are included.

### Freshwater tetrapods (first pass)

- Convergence bins: `thesis/convergence/output_occ_ecospace_tetrapoda_freshwater_min3_pairs15/timebin_metrics.csv`
- Robustness summary:
  - `thesis/synthesis/output_convergence_sampling_autocorr_tetrapoda_freshwater_v1/summary.md`

Outcome: strong correlation under time-only control (likely confounded); **drops** after adding sampling controls.

### Mammalia-only (terrestrial)

We can compute bins, but with strict pair-count thresholds we only got a handful of rich bins (mostly ≤60 Ma). This likely needs a
pair-level model rather than bin-level residual averages.

## Next upgrades (if we want “beyond marine” to be convincing)

1) Fit **pair-level** models (mixed / hierarchical) for terrestrial tetrapods or mammals to use the full pairwise sample sizes
   (instead of n≈15–20 bins).
2) Try alternative spatial binning (larger grid cells) to increase per-bin locality counts in older terrestrial bins.
3) Add an independent terrestrial sampling proxy (Macrostrat is NA-heavy; consider region-specific rock-area curves or focus on
   Cenozoic with better constraints).

