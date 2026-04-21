# Do volatile climates make ecosystems predictable?

## The question

When the tape of life plays out in different oceans, does it produce different ecosystems — or does the environment force them toward the same ecological solutions regardless of which species are present?

Gould argued that history dominates: replay evolution and you get a different outcome each time. Conway Morris countered that convergence is pervasive: natural selection channels life into a limited set of functional designs. This debate has mostly played out at the level of individual lineages. We asked whether it applies to *entire marine communities* across the Phanerozoic.

## What we did

We measured how functionally similar geographically separated marine regions (10° paleocoordinate grid cells) are, after accounting for whatever taxonomic overlap they share. The idea: if two distant regions share many of the same genera, they'll obviously have similar ecological roles. The interesting question is whether they're *more* functionally similar than their shared taxonomy would predict — whether different species are doing the same jobs.

We call this residual "functional excess similarity." We computed it across ~2 million fossil occurrences from the Paleobiology Database, spanning the last 540 million years in 40 time bins, using genus-level ecological role assignments (diet, motility, life habit). Then we tested whether this metric tracks an independent measure of climate volatility from CESM paleoclimate simulations.

## What we found

**The central result.** When climate shifts rapidly between 10 Myr intervals, distant marine regions converge on more similar ecological role mixtures — even when they share few or no species (partial *r* = 0.38; block bootstrap *p* = 0.020; controlling for time, sampling structure, and provinciality).

**How it works.** The effect is a "baseline shift." Volatility doesn't change how tightly function tracks taxonomy when regions *do* share species. Instead, it raises the floor — the minimum functional similarity between regions that share almost nothing taxonomically. Under volatile climates, even taxonomically alien regions end up with similar job portfolios.

**Where it's strongest.** The signal concentrates in the Mesozoic (*r* = 0.53), despite the Mesozoic having *lower* average volatility than the Paleozoic. This rules out a simple dosage explanation and points toward an interaction between environmental forcing and the maturity of marine ecospace occupation.

**Which roles shift.** Under volatile climates, stationary suspension feeders consistently expand (top 6 expanding roles are all suspension feeders) while fast-moving predators contract (largest contraction: fast-moving low-level epifaunal carnivores, Δ = −0.053). However, this role-level shift does not *explain* the convergence signal — the convergence is spatial homogenisation of the overall role mix across regions, not a global shift toward any single strategy.

**What survives scrutiny.** The sign is stable across all 40 leave-one-out samples. Block bootstrap *p*-values range from 0.020 to 0.029. OLS with autocorrelation-robust standard errors gives *p* = 0.037. The most conservative time-series model (SARIMAX with AR errors) is marginal (*p* = 0.079).

## Honest caveats

This is a real pattern in the data, but its interpretation carries important caveats.

**Ecospace annotation quality.** PBDB trait coverage correlates with the convergence metric at *r* = 0.90. Most of that correlation is a shared time trend (older bins have better coverage for the same reason they have higher convergence: Paleozoic marine faunas are dominated by well-studied groups). But marine-specific coverage retains a partial correlation of 0.37 with convergence after removing the time trend. Adding coverage as a control reduces the volatility effect from *r* = 0.38 to *r* = 0.33 — it survives the block bootstrap (*p* = 0.047) but not the exact circular-shift test (*p* = 0.10). Annotation quality partially confounds the result.

**Clade restriction.** When we restrict to individually well-annotated clades (brachiopods, bivalves, gastropods), the signal disappears entirely (*r* = −0.13). This is either because convergence is a genuinely cross-clade phenomenon that requires multiple lineages to manifest, or because the signal depends on mixing differentially annotated taxa. We can't currently distinguish these.

**Small N.** With 40 time bins, we're pushing the limits of what time-series inference can do. The exact circular-shift test has a resolution floor of *p* = 0.025, and the SARIMAX model — the most conservative parametric approach — doesn't clear α = 0.05.

## What it means (if it holds up)

Volatile climates appear to reduce the historical contingency of marine community structure. Under rapid environmental change, the identity of the species present matters less than the set of ecological strategies the environment permits. Different evolutionary lineages independently converge on the same limited menu of viable roles.

This is Conway Morris's convergence argument scaled up from organisms to ecosystems — and modulated by the environment. Stable climates allow more functionally idiosyncratic regional solutions. Volatile climates homogenise them.

The Phanerozoic record offers a 540-million-year empirical baseline for a prediction that may be uncomfortably relevant: under rapid anthropogenic climate change, marine ecosystems may lose taxonomic distinctiveness while converging on a narrower set of ecological strategies. The jobs will still get done — but by a smaller, more predictable cast of characters.
