# Stored results (exploratory): body-size structure vs biogeographic stability

Hypothesis tested (high novelty, exploratory): **longer biogeographic stability is associated with a stronger “missing middle sizes” pattern** (i.e., more bimodal body-size distributions) in dinosaurs.

This folder tracks the writeups + analysis code. Large per-run artifacts (tables, caches, most plots) are generated into `output*`
folders but are not tracked.

## Primary run (stored)

Command:

```bash
python thesis/body_size_stability/run_analysis.py \
  --out thesis/body_size_stability/output \
  --time-bin-myr 10 \
  --grid-deg 10 \
  --permutations 10000
```

Summary: `thesis/body_size_stability/output/summary.md`

Key correlations (time bins with stability defined: `n=9`; gap metric defined in `n=7` bins):

| Exclude Avialae | Mass | corr(stability,bimodality) | perm-p | corr(stability,gap_ratio) | perm-p |
|---:|---:|---:|---:|---:|---:|
| 0 | mass1 | +0.464 | 0.214 | -0.882 | 0.0499 |
| 0 | mass2 | +0.508 | 0.175 | -0.878 | 0.0208 |
| 1 | mass1 | +0.557 | 0.113 | -0.394 | 0.372 |
| 1 | mass2 | +0.588 | 0.0945 | -0.364 | 0.426 |

Interpretation (very guarded): at 10 Myr resolution, there is **consistent positive association** between the stability proxy and a bimodality heuristic, and a **strong negative association** between stability and the “gap ratio” missing-middle heuristic when Avialae are included.

## Sensitivity runs (stored)

### Spatial resolution (grid size), time bin fixed at 10 Myr

Grid 5°:

```bash
python thesis/body_size_stability/run_analysis.py \
  --out thesis/body_size_stability/output_bin10_grid5 \
  --time-bin-myr 10 --grid-deg 5 --permutations 10000
```

Summary: `thesis/body_size_stability/output_bin10_grid5/summary.md`

- Bimodality correlations strengthen and become significant in some variants (e.g., exclude Avialae + `mass2`: `r=0.675`, `p=0.047`).
- Gap-ratio correlations strengthen for “include Avialae” variants (e.g., `mass2`: `r=-0.960`, `p=0.0054`; `n=7` bins for gap metric).

Grid 15°:

```bash
python thesis/body_size_stability/run_analysis.py \
  --out thesis/body_size_stability/output_bin10_grid15 \
  --time-bin-myr 10 --grid-deg 15 --permutations 10000
```

Summary: `thesis/body_size_stability/output_bin10_grid15/summary.md`

- Bimodality correlations remain positive and become significant for `mass2` in both Avialae include/exclude runs (e.g., exclude Avialae + `mass2`: `r=0.723`, `p=0.026`).
- Gap-ratio correlations weaken (not significant).

### Temporal resolution (time-bin width)

Time bin 20 Myr (grid 10°):

```bash
python thesis/body_size_stability/run_analysis.py \
  --out thesis/body_size_stability/output_bin20_grid10 \
  --time-bin-myr 20 --grid-deg 10 --permutations 10000
```

Summary: `thesis/body_size_stability/output_bin20_grid10/summary.md`

- Correlations largely disappear (near-zero `r`, large permutation p-values).
- Caveat: fewer bins (`n=6–7`) and more averaging.

## Notes for future “publication-grade” work

- The stability proxy is **PBDB-occurrence-based** (sampling sensitive); publication-grade inference should include sampling controls and/or an independent plate/climate stability series.
- Current inference is correlation-only with very small `n` (time bins); treat as a **hypothesis generator**.

## Independent forcing test (non-PBDB Earth-system series)

To move beyond a PBDB-derived stability proxy, we downloaded an **independent** 10 Myr-sampled CESM snapshot dataset (Li et al. 2022; Scientific Data; figshare `10.6084/m9.figshare.19920662.v1`) and derived volatility time series:

- Dataset + derivation: `thesis/earth_system/climate_540myr/README.md`
- Derived time series: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`
- Body-size × Earth-system merge + tests: `thesis/body_size_stability/output_independent_stability/analysis_results.json`

Key signal (10 Myr bins; very small `n`):

- When **Avialae are included**, the “gap ratio” missing-middle heuristic (`gap_ratio_hist`; higher = *less* missing-middle) correlates **positively** with temperature volatility:
  - `corr(delta_from_prev_T_global_abs, gap_ratio_hist)` ≈ `+0.81` (perm‑p ≈ `0.019`, `n=8`) for `mass1`
  - `corr(delta_from_prev_T_global_abs, gap_ratio_hist)` ≈ `+0.89` (perm‑p ≈ `0.0025`, `n=8`) for `mass2`

Interpretation (guarded): across the limited dinosaur time window sampled here, **higher climate volatility is associated with a weaker missing‑middle signature**, consistent with the idea that “stability” can enable stronger separation into small/large size modes.
