# Blog assets

Charts generated from the paleontology convergence analysis, ready to drop into the blog post.

## Suggested ordering in the post

| File | Use as | Caption suggestion |
|------|--------|-------------------|
| `00_social_card.png` | Social share card / OG image | (no caption needed) |
| `01_hero_540myr_convergence.png` | **Title image** (top of post) | "Functional convergence between distant ocean regions across 540 million years. Each dot is one 10-million-year time bin; marker size shows climate volatility. The Paleozoic ceiling, Mesozoic transition, and Cenozoic floor are visible at a glance." |
| `02_volatility_vs_convergence_by_era.png` | After the "turns out, this is true!" section | "When you split the 540-million-year time series by era, the volatility-convergence correlation almost entirely lives in the Mesozoic. The Permian-Triassic boundary (circled) is the single most influential point." |
| `06_baseline_shift_mechanism.png` | After the ceiling/floor explanation | "Left: the mechanism. Volatility raises the floor of functional similarity between regions that share no species, without changing how tightly function tracks taxonomy when species are shared. Right: why this only manifests in the Mesozoic — Paleozoic ecosystems are already at the ceiling, Cenozoic ecosystems are at the floor, only the Mesozoic spans the transition zone where volatility has room to operate." |
| `03_functional_fingerprint.png` | After the "sit and filter" section | "Under high-volatility climates, suspension feeders consistently expand and fast-moving carnivores contract. The individual roles shift the way I predicted, but the shift doesn't drive the convergence itself — convergence is spatial homogenization of the entire mix, not a global swap from one strategy to another." |
| `04_climate_not_continents.png` | After the tectonic plates correction | "I originally thought continental rearrangement drove convergence. The data disagreed. Climate variables (temperature change, precipitation change) correlate with convergence; paleogeographic variables (land area change, coastline change) do not. The plates matter because they cause climate volatility, not because of their geography per se." |
| `05_modern_analog.png` | At the "current warming" closing | "Distribution of climate volatility across all 40 Phanerozoic time bins. Anthropogenic warming rates put us in the top 10% of anything Earth has seen in the last 540 million years." |

## Regenerate

```bash
uv run python blog_assets/make_charts.py
```

All charts use the same color scheme:
- Paleozoic: green (#2E7D5B)
- Mesozoic: red (#C0392B)
- Cenozoic: blue (#1F5F8B)
