# Paleontology Analytics

Reproducible analysis of Phanerozoic marine functional convergence and its relationship to climate volatility, built on the Paleobiology Database and CESM paleoclimate simulations.

The headline result: when climate shifts rapidly between 10 Myr intervals, geographically distant marine provinces converge on more similar ecological role mixtures — even when they share few or no species. Volatile climates compress the functional structure of marine ecosystems.

## Quick links

| What | Where |
|------|-------|
| Shareable findings summary | [`thesis/FINDINGS_SUMMARY.md`](thesis/FINDINGS_SUMMARY.md) |
| Full technical report | [`thesis/synthesis/FINAL_REPORT.md`](thesis/synthesis/FINAL_REPORT.md) |
| Project roadmap | [`thesis/README.md`](thesis/README.md) |
| Reproduce everything | `python thesis/run_all.py` |

## Repository layout

```
paleontology_analytics/
├── data/                    # Raw + processed datasets (gitignored)
├── src/                     # Shared acquisition/normalization code
│   ├── acquisition/         #   PBDB download helpers
│   └── normalization/       #   Data cleaning + schema
├── thesis/                  # All research code, analysis, and writing
│   ├── convergence/         #   Core convergence pipeline
│   ├── synthesis/           #   Robustness, sensitivity, inference scripts
│   ├── manuscript_*/        #   Paper draft + supplement (gitignored)
│   ├── earth_system/        #   CESM climate forcing derivation
│   ├── body_size_stability/ #   Dinosaur body-size track (secondary)
│   ├── geographic_portfolio/#   Extinction survivorship track (secondary)
│   └── run_all.py           #   One-button reproduction
├── tests/                   # Regression tests
├── dashboard/               # Static data dashboard
└── requirements.txt         # Python dependencies
```

## Getting started

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure data exists (PBDB + CESM; see thesis/README.md for details)
# Then run the full pipeline:
python thesis/run_all.py
```

## Development gates

- `make check` — Ruff linting for `src/`, `thesis/`, and `tests/`
- `make test` — pytest regression suite
- `make all` — check then test
- `make deps-audit` — advisory pip-audit

## License

Apache-2.0 (see `LICENSE`).
