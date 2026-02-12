# Paleontology Analytics

Does climate volatility make ecosystems predictable? This project tests whether rapid climate change forces geographically distant marine communities to converge on similar ecological structures — the Gould-vs-Conway-Morris debate, scaled up from organisms to entire ecosystems and played out across 540 million years of fossil record.

Built on ~2 million occurrences from the Paleobiology Database and CESM paleoclimate simulations (Li et al. 2022).

## What we tested

Four research tracks, one core and three secondary:

1. **Marine functional convergence** — Do volatile climates force taxonomically distinct marine provinces to converge on similar ecological roles? (core; `thesis/convergence/` and `thesis/synthesis/`)
2. **Dinosaur body-size structure** — Does climate stability enable the "missing middle" in dinosaur body-size distributions? (`thesis/body_size_stability/`)
3. **Geographic portfolio and mass extinctions** — Does range *configuration* — not just size — predict which genera survive major crises? (`thesis/geographic_portfolio/`)
4. **Paleobiotic velocity** — Does how far a genus shifts its geographic centroid between time bins predict extinction risk? (`thesis/paleobiotic_velocity/`)

## What we found

### The main result

When climate shifts rapidly between 10 Myr intervals, geographically distant marine provinces converge on more similar ecological role mixtures — even when they share few or no species (partial r = 0.38; block bootstrap p = 0.02; controlling for time, sampling, and provinciality).

The mechanism is a "baseline shift": volatility raises the *floor* of functional similarity between provinces that share almost no genera, rather than changing how tightly function tracks taxonomy when species are shared. Under volatile climates, different evolutionary lineages independently converge on the same limited menu of viable ecological roles.

The signal concentrates in the Mesozoic (r = 0.53), despite the Mesozoic having lower average volatility than the Paleozoic. The reason remains an open question.

### Scorecard

| Track | Result | Strength |
|-------|--------|----------|
| Marine convergence | Positive: r = 0.38, bootstrap p = 0.02 | Moderate — survives most tests, marginal under SARIMAX |
| Baseline-shift mechanism | Volatility raises the floor, not the slope | Strong — clear mechanistic signature |
| Mesozoic concentration | Signal is a Mesozoic phenomenon | Descriptive — explanation open |
| Dinosaur body size vs volatility | Higher volatility weakens the missing-middle (r = 0.85) | Weak — n = 8 bins |
| Geographic portfolio (early crises) | Multi-province ranges favoured at end-Ordovician and Late Devonian | Weak — event-dependent, coordinate-sensitive |
| Paleobiotic velocity | Mobility OR < 1, but ΔAUC ≈ 0.001 | Negligible — fails modern-coordinate negative control |

### Dead ends

These hypotheses were tested and produced no signal:

- **PBDB turnover as forcing proxy** — goes the wrong way after removing the time trend
- **Terrestrial vertebrate convergence** — wrong sign, not significant (underpowered)
- **Clade-restricted convergence** — signal disappears when limited to well-annotated clades alone
- **Low-energy role mediation** — a "sit-and-filter" composite doesn't track volatility
- **Role interchangeability (mutual information)** — no link between volatility and how taxonomically interchangeable roles are
- **Spatial coherence vs magnitude** — can't separate from raw volatility magnitude in this dataset
- **Equator-crossing as extinction buffer** — inconsistent across events

## Caveats

- **Annotation quality** is the biggest confound. PBDB ecospace coverage correlates at r = 0.90 with the convergence metric (mostly a shared time trend, but marine-specific coverage retains partial r = 0.37 after detrending). Adding coverage as a control attenuates the volatility effect from r = 0.38 to r = 0.33.
- **Clade restriction** kills the signal. Unclear whether convergence genuinely requires cross-clade mixing or whether differential annotation quality across clades is doing the work.
- **Small N.** 40 time bins is near the floor for time-series inference. The most conservative parametric model (SARIMAX AR(0)) gives p = 0.079.
- **Grid sensitivity.** The signal is positive at 10° and 15° but vanishes at 20°.

## Dig deeper

| What | Where |
|------|-------|
| Findings summary (shareable) | [`thesis/FINDINGS_SUMMARY.md`](thesis/FINDINGS_SUMMARY.md) |
| Full technical report | [`thesis/synthesis/FINAL_REPORT.md`](thesis/synthesis/FINAL_REPORT.md) |
| Research log (every hypothesis tested) | [`thesis/RESEARCH_LOG.md`](thesis/RESEARCH_LOG.md) |
| Project roadmap and folder guide | [`thesis/README.md`](thesis/README.md) |

<details>
<summary>Reproduce the results</summary>

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure data exists (PBDB + CESM; see thesis/README.md for sources)
# Then run the full pipeline:
python thesis/run_all.py

# Or selectively:
python thesis/run_all.py --skip-core        # skip data-heavy convergence recomputation
python thesis/run_all.py --only-hardening   # only sensitivity/robustness scripts
```

</details>

<details>
<summary>Repository layout</summary>

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

</details>

## Development

- `make check` — Ruff linting
- `make test` — pytest regression suite
- `make all` — check then test
- `make deps-audit` — advisory pip-audit

## License

Apache-2.0 (see `LICENSE`).
