# Convergence analyses (PBDB ecospace × paleogeography)

This folder contains analysis code + writeups aimed at testing three related, high-level convergence questions using:
- PBDB occurrences from this repo: `data/processed/merged_occurrences.parquet`
- PBDB **taxon ecospace** annotations (diet/motility/life habit/environment) fetched via PBDB API
- (Optional) independent Earth-system time series (climate/paleogeography) downloaded into `thesis/earth_system/`

## Questions (as requested)

1) **Functional convergence without taxonomic convergence**
   - After “perturbations”, do regions become *functionally* similar even when they are *taxonomically* different?

2) **Convergence vs fragmentation (provinciality)**
   - Is functional convergence stronger when biogeographic provinces are more fragmented / distinct?

3) **Convergence peaks during volatility (not stability)**
   - Is functional convergence strongest during globally volatile intervals (vs stable ones)?

These are explored first with PBDB-derived turnover/provinciality/volatility metrics, and then (when available) with an **independent** climate/paleogeography forcing series.

Start here:
- `thesis/convergence/RESULTS.md`
- `thesis/synthesis/FINAL_REPORT.md` (ties the robustness stack together)

## Run

```bash
python thesis/convergence/run_convergence_analysis.py \
  --out thesis/convergence/output_run \
  --time-bin-myr 10 \
  --grid-deg 10 \
  --min-occ-per-genus 5 \
  --min-genera-per-region 25 \
  --max-pairs-per-bin 30000 \
  --permutations 20000
```

## Outputs

- `output_run/summary.md`: short interpretation + caveats.
- `output_run/*.csv`: per-bin metrics + sampled pairwise similarities (generated; not tracked).
- `output_run/**/figures/*.png`: generated plots (not tracked).

## Canonical stored run

The current synthesis defaults point at the full‑PBDB run in:
- `thesis/convergence/output_v3_fullpbdb/`

Older exploratory runs are archived under:
- `thesis/_archive/convergence/` (local-only; not tracked).

## Notes / caveats

- PBDB ecospace is an expert-coded annotation; missingness is non-random.
- PBDB occurrences encode sampling/rock/collection effects; results are **hypothesis-generating** unless controlled with independent sampling proxies and/or independent forcing series.
