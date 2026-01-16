# Archived pilot: genus centroid “paleovelocity”

This folder is kept for historical context: an early pilot run (writeup + a couple of figures) that explored a centroid-velocity proxy and a
simple extinction association. The current, bias-aware pipeline lives at `thesis/paleobiotic_velocity/`.

## Reproduce (optional)

From the repo root:

```bash
python thesis/archive/paleovelocity_pilot/code/paleovelocity.py \
  --data data/processed/merged_occurrences.parquet \
  --out thesis/archive/paleovelocity_pilot/output
```

Outputs are written under `thesis/archive/paleovelocity_pilot/output/` and are gitignored by default.

## Notes

- `data/raw/` and `data/processed/` are gitignored (large datasets).
- This pilot uses a simplified terminal-bin framing; use `thesis/paleobiotic_velocity/` for the more careful survival-model treatment and
  negative controls.
