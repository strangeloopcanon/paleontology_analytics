# Research log: hypotheses, experiments, outcomes

A record of what was tested, what worked, and what didn't. Organised by research track; within each track, experiments are roughly chronological.

---

## Track 1: Marine functional convergence

The question that became the paper. Do volatile climates force taxonomically distinct marine provinces to converge on similar ecological roles?

### Does functional convergence follow perturbations? (PBDB turnover proxy)

**Hypothesis.** After episodes of high genus turnover, distant provinces should become more functionally similar than expected from shared taxonomy.

**Test.** Correlated PBDB-derived global turnover (Jaccard between adjacent 10 Myr bins) with functional excess similarity.
Script: `convergence/run_convergence_analysis.py`.

**Outcome.** Raw r = 0.10 (p = 0.52); after controlling for time r = −0.33 (p = 0.04). The relationship goes the *wrong way* once you remove the time trend.

**Verdict.** Dead end. PBDB turnover is a poor proxy for environmental forcing — it conflates sampling artifacts with genuine biological signal.

### Does an independent climate forcing series do better?

**Hypothesis.** Higher climate volatility (from CESM simulations, not from PBDB itself) is associated with greater functional excess similarity.

**Test.** Merged PBDB convergence bins with Li et al. (2022) CESM paleoclimate snapshots. Volatility = mean absolute temperature change between consecutive 10 Myr equilibria.
Script: `convergence/test_independent_forcing.py`.

**Outcome.** Field-mean temperature volatility: partial r = 0.50 controlling for time (p = 0.003, n = 33). Much stronger than the PBDB turnover proxy.

**Verdict.** Positive. The independent forcing series uncoupled the question from PBDB sampling artifacts. This became the core of the project.

### Does it survive sampling controls?

**Hypothesis.** The volatility-convergence association could be a sampling artifact: more fossiliferous bins might show both higher apparent volatility and more functional overlap.

**Test.** Added PBDB sampling proxies (marine collections, occurrences, localities), Macrostrat rock-record proxies (section counts, column areas), compressed via PCA to avoid collinearity. Used circular-shift nulls to handle autocorrelation.
Script: `synthesis/robust_convergence_sampling_autocorr.py`.

**Outcome.** Primary specification (time + sampling PCA PC1+PC2 + provinciality): partial r = 0.38. Block bootstrap p = 0.020. OLS + Newey-West HAC p = 0.037. SARIMAX AR(0) p = 0.079.

**Verdict.** Positive, with a caveat. The non-parametric tests and HAC-corrected OLS support it. The most conservative parametric time-series model (SARIMAX) is marginal.

### Is it a baseline shift or a slope change?

**Hypothesis.** Volatility raises the *floor* of functional similarity between taxonomically distinct provinces (baseline shift), rather than changing how tightly function tracks taxonomy when taxa are shared (slope change).

**Test.** Pair-level regression of functional similarity on taxonomic similarity, volatility, and their interaction, with cluster-robust standard errors.
Script: `synthesis/pair_level_convergence_model.py`.

**Outcome.** Volatility raises the intercept (β = 0.019, cluster p = 0.002). The interaction term (slope change) is also significant but does not survive circular-shift nulls (p = 0.15). Per-bin intercept positively correlates with volatility.

**Verdict.** Positive. The baseline-shift signature is the mechanistic headline: volatile climates make provinces functionally similar even when they share no genera.

### Which ecospace axes drive it?

**Test.** Decomposed functional similarity into diet, motility, and life habit axes.
Script: `convergence/run_role_decomposition.py`.

**Outcome.** All three axes show convergence with volatility. Life habit is strongest (partial r = 0.55, p < 0.001), followed by diet (r = 0.40) and motility (r = 0.38).

**Verdict.** All axes contribute; life habit dominates.

### What specific roles change?

**Test.** Measured per-role geographic ubiquity and within-locality share against volatility with sampling controls.
Script: `synthesis/role_job_drivers_volatility.py` (now archived).

**Outcome.** Suspension feeders and stationary organisms increase in volatile bins; fast-moving carnivores decrease. Individual role tests don't survive multiple-comparison correction, but the direction is consistent.

**Verdict.** Suggestive pattern toward "sit-and-filter" roles, but not statistically robust.

### Does a pre-registered low-energy composite mediate the effect?

**Hypothesis.** A composite "low-energy / sit-and-filter" index should track volatility and mediate the convergence signal.

**Test.** Defined a composite from ecospace axes (low-energy diet/motility/habitat minus high-energy). Tested against volatility with circular shifts and as a mediator in the pair-level model.
Script: `synthesis/low_energy_index_mediation.py` (now archived).

**Outcome.** The index does not track volatility under circular shifts (p ≈ 0.20) and attenuates the volatility coefficient by only ~5%.

**Verdict.** Dead end. The convergence signal is not explained by a global shift toward low-energy composition. The mechanism may be spatial homogenisation rather than a change in the overall average role mix.

### Does spatial coherence of forcing matter more than magnitude?

**Hypothesis.** Spatially coherent climate change (uniform direction across regions) might drive convergence more than raw magnitude.

**Test.** Derived multiple coherence/patchiness metrics from CESM ΔT fields. Tested independently and alongside magnitude.
Script: `synthesis/coherence_beats_volatility.py` (now archived).

**Outcome.** Coherence metrics correlate with convergence, but they're strongly collinear with magnitude. No independent explanatory power.

**Verdict.** Dead end. Can't separate coherence from magnitude in this dataset.

### Are roles less clade-specific in volatile climates? (Mutual information)

**Hypothesis.** Volatility makes ecological roles taxonomically interchangeable: knowing the role tells you less about which clade is filling it.

**Test.** Computed normalised mutual information between genus (or family/order) and role per bin. Tested against volatility with sampling controls.
Script: `synthesis/role_interchangeability_mi.py` (now archived).

**Outcome.** No detectable association between volatility and MI for any of the primary metrics.

**Verdict.** Dead end.

### Is the signal concentrated in one era?

**Test.** Split the analysis by Paleozoic, Mesozoic, and Cenozoic.
Script: `synthesis/era_heterogeneity_investigation.py`.

**Outcome.** Mesozoic (16 bins): r = 0.53, p = 0.05. Paleozoic (17 bins): r = −0.11. Cenozoic (7 bins): r ≈ 0. The Mesozoic has *lower* mean volatility than the Paleozoic, ruling out a dosage explanation.

**Verdict.** Confirmed. The signal is a Mesozoic phenomenon. The reason remains an open question (ecospace maturity? paleogeography?).

### Does it survive clade restriction?

**Test.** Reran convergence using only well-annotated clades (Brachiopoda, Bivalvia, Gastropoda).
Script: `synthesis/clade_restriction_test.py`.

**Outcome.** Brachiopoda (18 bins): r = −0.09. Combined (32 bins): r = −0.13. Signal disappears.

**Verdict.** Negative. The signal either requires cross-clade mixing to emerge, or it requires mixing differentially annotated taxa. A genuine limitation.

### Does annotation quality confound the result?

**Test.** Characterised PBDB ecospace annotation completeness per bin. Computed partial correlations controlling for time. Added coverage as a control.
Script: `synthesis/ecospace_missingness_diagnostic.py`, `synthesis/robustness_battery.py`.

**Outcome.** Raw coverage-convergence correlation: r = 0.90. Generic coverage is fully absorbed by the time trend (partial r = −0.03). Marine-specific coverage retains partial r = 0.37 after time. Adding coverage as a control: volatility r drops to 0.33, block bootstrap p = 0.047.

**Verdict.** Partially confounded. The signal survives but is attenuated. Annotation quality is the single most important caveat.

### Does it extend to terrestrial vertebrates?

**Test.** Applied the same pipeline to PBDB terrestrial tetrapods.
Script: `synthesis/terrestrial_convergence_pilot.py`.

**Outcome.** 3663 genera, 11 qualifying bins: r = −0.40 (p = 0.22). Wrong sign, not significant, low power.

**Verdict.** Dead end. No evidence for terrestrial convergence (though the test is underpowered).

### Does it depend on grid resolution?

**Test.** Reran at 10°, 15°, 20° grid resolutions.
Script: `synthesis/grid_sensitivity.py`.

**Outcome.** 10°: r = 0.23 (p = 0.16). 15°: r = 0.23 (p = 0.06). 20°: r = 0.07 (p = 0.82). Signal vanishes at coarse resolution.

**Verdict.** Scale-dependent. The spatial grain matters; very coarse grids wash out the pattern.

---

## Track 2: Dinosaur body-size structure

Secondary track. Does climate volatility reshape the distribution of dinosaur body sizes?

### Does biogeographic stability predict bimodal body-size distributions?

**Hypothesis.** Longer periods of geographic stability allow niche partitioning to develop, producing a "missing middle" (barbell) in dinosaur body-size distributions.

**Test.** Computed per-bin size distribution metrics (bimodality coefficient, gap ratio) from Benson et al. (2014) mass estimates. Correlated with a PBDB-derived stability proxy.
Script: `body_size_stability/run_analysis.py`.

**Outcome.** Gap ratio vs stability: r = −0.88 (p = 0.05, n = 9, with Avialae, mass2). Bimodality coefficient: positive but not significant.

**Verdict.** Positive, but n = 8–9 bins. Fragile.

### Does independent climate volatility predict body-size structure?

**Test.** Merged body-mass bins with CESM volatility series.
Script: `body_size_stability/test_independent_stability.py`.

**Outcome.** Gap ratio vs field-mean volatility: r = +0.85 (p = 0.008, n = 8). Higher volatility → weaker missing-middle.

**Verdict.** Positive. Consistent with the stability hypothesis via independent forcing. But n = 8 is tiny.

---

## Track 3: Geographic portfolio and mass extinction survivorship

Does the spatial configuration of a genus's range — not just its size — predict whether it survives mass extinctions?

### Does connectedness (single-core vs multi-core range) predict survivorship?

**Test.** Logistic regression of genus survivorship on pre-event geographic metrics (largest component fraction, total range, sampling proxies) across four Phanerozoic crises.
Script: `geographic_portfolio/run_event_portfolio_analysis.py`.

**Outcome.** End-Ordovician and Late Devonian: multi-core ranges favoured (OR < 1 for connectedness). End-Permian: weak/non-robust. End-Triassic: mixed.

**Verdict.** Event-dependent. Works for early Phanerozoic crises, not universally.

### Does portfolio entropy (range evenness) help?

**Test.** Shannon entropy of component-size shares as a predictor.
Script: `geographic_portfolio/run_additional_hypotheses.py`.

**Outcome.** Positive for end-Ordovician (OR ≈ 2.25) and Late Devonian (OR ≈ 1.85). Negative or null for end-Permian and end-Triassic.

**Verdict.** Same pattern: early crises only.

### Does equator-crossing help?

**Test.** Whether genera spanning both hemispheres survive better.

**Outcome.** Inconsistent across events; often negative.

**Verdict.** Dead end.

### Does spatial dispersion (beyond range size) help?

**Test.** Log mean distance from occupied cells to centroid.

**Outcome.** Positive for end-Permian (OR ≈ 1.64). Negative for end-Ordovician. Mixed otherwise.

**Verdict.** Event-dependent, no universal pattern.

---

## Track 4: Paleobiotic velocity

Does how far a genus moves its geographic centroid between time bins predict its extinction risk?

### Does apparent mobility reduce extinction hazard? (Pilot)

**Test.** Genus centroid displacement rates from paleocoordinates; terminal extinction logistic model.
Script: `archive/paleovelocity_pilot/paleovelocity.py`.

**Outcome.** Mobile genera have lower terminal extinction rates (OR ≈ 0.86 per SD). AUC ≈ 0.69.

**Verdict.** Positive in the pilot.

### Does it survive rigorous discrete-time survival modelling?

**Test.** Discrete-time survival model with time-bin fixed effects, occurrence-weighted and locality-weighted centroids, modern-coordinate negative control.
Script: `paleobiotic_velocity/run_pipeline.py`.

**Outcome.** AUC (full model) ≈ 0.798 vs AUC (baseline) ≈ 0.796. Incremental ΔAUC ≈ 0.001. Modern-coordinate negative control gives similar ΔAUC.

**Verdict.** Effectively dead. The signal is real but trivially small, and the modern-coordinate control suggests it may be a sampling artifact rather than biology.

---

## Scorecard

| Track | Main finding | Strength |
|-------|-------------|----------|
| Marine convergence | Volatile climates compress functional structure (r = 0.38, bootstrap p = 0.02) | Moderate — survives most tests but not SARIMAX or clade restriction |
| Baseline-shift mechanism | Volatility raises the floor, not the slope | Strong — clear mechanistic signature |
| Mesozoic concentration | Signal is a Mesozoic phenomenon | Descriptive — explanation is open |
| Dinosaur body size | Volatility weakens the missing-middle (r = 0.85) | Weak — n = 8 bins |
| Geographic portfolio | Range configuration predicts early-Phanerozoic crisis survival | Weak — event-dependent, coordinate-sensitive |
| Paleobiotic velocity | Mobility → lower extinction risk | Negligible — trivial effect size, fails negative control |
| Terrestrial convergence | No signal | Dead end |
| Clade-restricted convergence | No signal | Dead end |
| Low-energy mediation | No signal | Dead end |
| Role interchangeability (MI) | No signal | Dead end |
| Coherence > magnitude | Can't separate from magnitude | Dead end |
