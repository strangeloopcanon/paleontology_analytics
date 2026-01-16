# Final wrap (current state): volatility ↔ convergence, and dinosaur size-structure

This is a “put a bow on it” snapshot of what we have, what we fixed, what can still break the claims, and what the next
publication‑grade steps should be.

Note on paths: most artifacts under `data/` and `output*/` are generated locally and gitignored (e.g., `*.csv`, `*.parquet`, `*.json`, `*.png`). The tracked “paper trail” is the `summary.md` files plus curated figures; see `thesis/run_all.py` to reproduce outputs.

## 1) Are we using the full PBDB?

Yes (for Cambrian→Holocene, which matches the 540 Myr climate series we use).

What was missing before:
- The project’s PBDB acquisition default is **Cambrian → Cretaceous** (`src/acquisition/pbdb.py`), so the repo’s processed tables
  originally had **no Cenozoic**.
- Also, the repo’s canonical normalization schema drops PBDB fields that are critical for sampling controls (e.g., `collection_no`),
  even though the raw CSV has them.

What we did (now complete):
- Downloaded the Cenozoic slice **reliably** using a paged downloader (avoids truncated `limit=all` streams):
  - Script: `thesis/pbdb/download_pbdb_occurrences_paged.py`
  - Raw output: `data/raw/pbdb_occurrences_paleogene_holocene_paged.csv` (873,054 occurrences; `mid_ma` ≈ 0–65.7 Ma)
- Rebuilt the canonical and extended PBDB parquets:
  - Canonical: `data/processed/pbdb_occurrences.parquet` (1,973,558 PBDB occurrences; `mid_ma` ≈ 0–534.8 Ma)
  - Extended (retains `collection_no` etc): `data/processed/pbdb_occurrences_extended.parquet`
  - Merged: `data/processed/merged_occurrences.parquet`

Note:
- This is still not “literally everything PBDB ever” (e.g., pre-Cambrian), but it is the full span relevant to our 540 Myr climate
  forcing series.

## 2) What’s the best “insight” result right now?

### A) Marine functional convergence tracks climate volatility (and survives key robustness checks)

Claim (operational):
> When climate volatility is higher, distant marine provinces become **more functionally similar than expected from taxonomy**.

Core pipeline artifacts:
- Convergence bins (PBDB ecospace v2; full PBDB): `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`
- Independent forcing: `thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv`

Initial independent forcing test:
- `thesis/convergence/output_independent_forcing/summary.md`

Robustness upgrade (important):
- Added sampling proxies from `collection_no` (and an autocorrelation-aware p-value via circular shifts).
- Full-PBDB summary: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb/summary.md`
- Macrostrat sensitivity check (naive covariate stacking): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_v2/summary.md`
- Macrostrat + PBDB sampling index (PCA; fixes collinearity): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/summary.md`
- Pair-level regression (cluster-robust + circular-shift null; publication-oriented): `thesis/synthesis/output_pair_level_model_volatility_v1/summary.md`

Additional “coherence” upgrade (new knob):
- Derived multiple spatial-coherence / patchiness metrics from the CESM ΔT fields (not just magnitude).
- Tested whether coherence improves inference beyond magnitude:
  - `thesis/synthesis/output_coherence_beats_volatility/summary.md`
  - Key outcome: several coherence metrics correlate with convergence, but they are also strongly correlated with magnitude in this dataset; coherence does **not** add independent explanatory power once magnitude is included.

Key numbers (partial correlations; volatility vs convergence residuals):
- Controls = time only: `corr ≈ +0.409` (iid perm‑p ≈ `0.008`; circular‑shift p ≈ `0.026`)
- Controls = time + `log1p(n_localities)`: `corr ≈ +0.289` (iid perm‑p ≈ `0.071`; circular‑shift p ≈ `0.106`)
- Controls = time + locality + marine collections + marine occurrences: `corr ≈ +0.350` (iid perm‑p ≈ `0.027`; circular‑shift p ≈ `0.027`)
- Add provinciality too: `corr ≈ +0.358` (iid perm‑p ≈ `0.025`; circular‑shift p ≈ `0.028`)

Interpretation (guarded, but now much less fragile):
- This is no longer “just a time trend” or a trivial sampling artifact (we explicitly controlled for sampling proxies and used a
  time-series-aware null).
- It’s plausibly a real macroecological pattern: volatility pushes different regions toward similar “jobs” even if the species
  differ.

New mechanistic signature (strongly aligned with “different taxa, same jobs”):
- When we fit the per-bin relationship `functional_similarity ~ taxonomic_similarity`, forcing does **not** measurably change the slope,
  but it **does** raise the intercept (baseline functional similarity at low taxonomic similarity):
  - `thesis/synthesis/output_function_taxonomy_coupling_v3_binslopes/summary.md`
  - This is consistent with the “synchronized filter” story: even taxonomically distinct provinces converge on similar role mixes.

Additional mechanistic probe (new; did **not** support the “interchangeability increases” story):
- We tested whether volatility makes roles *less clade-specific* using mutual information between taxonomic groups (`family`/`order`) and roles
  (`diet|motility|life_habit`) on the **same bins + localities** as the main convergence pipeline.
  - Result: no detectable volatility association for the primary normalized metric (bin-level; time + sampling PCA + provinciality controls):
  - `thesis/synthesis/output_role_interchangeability_mi_v1/summary.md`

New interpretability pass (“what jobs change?”; sampling+autocorr-aware):
- We quantified two simple, interpretable category dynamics across bins:
  1) **Geographic ubiquity** = fraction of localities where a category appears at all.
  2) **Mean within-locality share** = average fraction of genera in a locality belonging to that category.
- Then we tested each against volatility with the same bin-level controls as the pair-level model (time + sampling PCA + provinciality),
  and used the same circular-shift null.
  - Output: `thesis/synthesis/output_role_jobs_volatility_v1/summary.md`

What it suggests (mechanistic story, still exploratory):
- Under higher volatility, **locality composition shifts toward “sit-and-filter” roles**:
  - Diet: **suspension feeder** mean share tends to increase (shift p≈0.075).
  - Motility: **stationary** mean share increases (shift p≈0.10) while **facultatively mobile** decreases (shift p≈0.025).
  - At the specific role-combination level, several **suspension feeder | stationary | epifaunal** roles rise, while some
    **fast-moving carnivore** roles fall (smallest shift p≈0.025, but these do **not** survive BH correction over many roles).
- So the best, honest interpretation is: volatility raises baseline functional similarity partly because it **compresses ecosystems toward a
  smaller set of robust, low-energy role mixtures**, and/or removes some high-energy/mobile predator roles from many localities.

Mechanism hardening attempt (pre-registered composite index; negative result but informative):
- We defined a single “low-energy / sit-and-filter” composite index from ecospace axes (low-energy diets/motility/life-habits minus high-energy ones),
  tested it vs volatility with the same circular-shift null, and tested mediation/attenuation in the pair-level model.
- Result: the index does **not** track volatility under circular shifts (p≈0.20) and does not materially attenuate the volatility coefficient (≈5%).
  - `thesis/synthesis/output_low_energy_index_mediation_v1/summary.md`
Interpretation: the convergence signal is unlikely to be explained purely by a *global mean* shift toward low-energy composition; the driver may be
more about **spatial homogenization/constraint** than about changing the overall average mix.

### B) Dinosaur “missing-middle” weakens under volatility (intriguing but small-n)

Claim (operational):
> More volatility → dinosaur size distributions look less “barbell” (the middle is more filled in).

Artifact:
- `thesis/body_size_stability/output_independent_stability/analysis_results.json`

Strength:
- The volatility ↔ gap metric is strong in the available bins (e.g., `corr ≈ +0.85` for the gap metric under one variant),
  but **n is tiny** (≈8 usable bins for that test). This is the biggest fragility.

Novelty positioning (so far):
- Dinosaur size distribution shape/modality is studied (e.g., O’Gorman & Hone 2012; cached PDF under `thesis/literature/pdfs/`),
  but I have not yet found work tying *missing-middle strength* to an **independent deep-time volatility series**.

## 3) Key reasons the claims could be wrong, and what we did about them

### (i) “This is just sampling/rock record bias”

Fixes already implemented (marine convergence):
- Added bin-level sampling proxies from PBDB **collections** and **occurrence counts** (marine‑classified) via
  `data/processed/pbdb_occurrences_extended.parquet`.
- The volatility→convergence association stays positive across control sets; it is statistically supported under PBDB sampling-proxy controls.
- Macrostrat rock-record proxies are strongly collinear with PBDB sampling proxies; naively stacking multiple Macrostrat covariates destabilizes the residualization.
- Using a **PCA sampling index** over PBDB+Macrostrat proxies resolves the collinearity: with controls `time + provinciality + sampling_PC1 + sampling_PC2`, the effect remains supported (circular-shift p ≈ 0.024 in `...macrostrat_pca_v1/...`).

Still missing (publication-grade):
- A single, principled sampling model (e.g., pair-level hierarchical model with regularization/priors) rather than bin-level residualization.
- Explicit hierarchical models that separate biological signal from sampling structure.

### (ii) “Time bins are autocorrelated; your p-values are wrong”

Fix implemented (marine convergence):
- Circular-shift null on residuals (preserves autocorrelation in time-ordered bins).
- Effect remains supported under shift p-values.

Still missing (publication-grade):
- Explicit time-series models (e.g., GLS/AR errors) or block bootstraps tuned to the series’ autocorrelation.

Update (now done, first pass):
- We added an explicit **bin-level time-series regression** with AR(1) errors (SARIMAX) and an OLS+HAC sensitivity check, plus a **pair-level mixed-effects** model (random bin effects).
  - `thesis/synthesis/output_time_series_hierarchical_models_v1/summary.md`
  - Key nuance: OLS/HAC supports `vol_z > 0` on bin-level excess similarity, while SARIMAX AR(1) is more conservative (vol term not supported in that spec). The pair-level mixed model supports `vol_z > 0`.
  - This is good “reviewer-proofing”: it explicitly shows what changes when we assume serially-correlated errors.

### (iii) “PBDB ecospace traits are incomplete/heterogeneous”

What we did:
- Role decomposition into diet/motility/life habit to check which axis drives the signal:
  - `thesis/convergence/output_role_decomposition/summary.md`

Still missing:
- Family/order-level ecospace fallback to reduce missingness.
- Sensitivity to alternative role codings and to ecospace missingness by time/clade.

### (iv) “You can’t unify dinosaurs with marine convergence”

True right now:
- PBDB ecospace terrestrial coverage is too thin to replicate the marine convergence pipeline for dinosaurs.
- The Mesozoic-only slice does not cleanly reproduce the same marine convergence–volatility signal in our bins, even though the
  dinosaur size-structure signal is strongest there.

So the unification is currently conceptual (“volatility disrupts stable structure, pushes convergence”), not a single tight,
same-domain empirical chain.

## 4) What’s next (best path to a publication-ready write-up)

**If the bar is publication-ready with real insight (not just method), the best bet is to center the write-up on the marine result** and
treat dinosaur size structure as an exploratory extension unless we can expand it.

Concrete next steps (in order):
1) Replace the current “macro covariate sensitivity check” with a publication-grade sampling model:
   - Macrostrat is now wired in (`data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv`) and the effect remains positive but weakens (borderline under circular shifts when adding a single Macrostrat covariate).
   - Done: Macrostrat+PBDB sampling index via PCA (stable under collinearity) in `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/summary.md`.
   - Next: move to a pair-level model (and add spatial standardization) so sampling control is part of a single inferential model rather than residualization.
2) Move from bin-level correlations to pair-level/hierarchical models:
   - Model pairwise functional similarity as a function of taxonomic similarity, volatility, time, sampling proxies, province
     structure; include random effects by bin and maybe by locality.
3) Probe heterogeneity explicitly (this is now a real clue, not a nuisance):
   - In our current bins, the volatility→convergence signal is strong in the **Mesozoic** and weak in the **Paleozoic** under within-era controls:
     `thesis/synthesis/output_subera_volatility_convergence/summary.md`
   - That suggests the next “mechanism chapter” is to ask *what differs across eras* (oxygen regime? connectivity? baseline ecospace filling? sampling structure?).
4) Terrestrial functional dataset acquisition:
   - Either build a curated trait map for tetrapods/dinosaurs (external datasets), or shift to a terrestrial clade with better
     functional annotation.
5) Dinosaur robustness:
   - Add more mass datasets / alternative size proxies; propagate uncertainty; increase bin count if possible.

## 5) Where to look

- Convergence vs volatility robustness (full PBDB): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb/summary.md`
- Convergence vs volatility robustness (+ Macrostrat): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_v2/summary.md`
- Convergence vs volatility robustness (+ Macrostrat, PCA sampling index): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/summary.md`
- Role/job interpretability (which ecospace categories shift with volatility): `thesis/synthesis/output_role_jobs_volatility_v1/summary.md`
- Low-energy composite index mediation attempt: `thesis/synthesis/output_low_energy_index_mediation_v1/summary.md`
- Explicit time-series + hierarchical inference (AR errors + mixed effects): `thesis/synthesis/output_time_series_hierarchical_models_v1/summary.md`
- Synthesis attempts + dinosaur alignment: `thesis/synthesis/output_volatility_filter_v4/summary.md`
- Dinosaur size results: `thesis/body_size_stability/RESULTS.md`
- Convergence results: `thesis/convergence/RESULTS.md`
- Novelty memo: `thesis/synthesis/NOVELTY_REVIEW.md`
- Draft paper + supplement: `thesis/manuscript_convergence_volatility/manuscript.md`, `thesis/manuscript_convergence_volatility/supplement.md`
- Reproduce everything: `python thesis/run_all.py`
