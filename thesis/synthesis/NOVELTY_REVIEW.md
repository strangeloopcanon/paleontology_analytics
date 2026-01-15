# Novelty + literature positioning (working memo)

This memo is a *first-pass* positioning document to answer “what’s new here?” relative to prior literature, and what would be
required to make the work publication-grade.

For an auto-built reading list (broad; needs manual triage), see:
- `thesis/literature/reading_lists/reading_list_openalex.md`

## What we actually found (stored results)

### A) Dinosaurs: “missing-middle” (barbell) varies with volatility

Empirical result (exploratory; small n bins):
- In Benson et al. (2014) dinosaur mass estimates binned at 10 Myr, the **gap metric** (`gap_ratio_hist`) increases with
  **climate volatility** (Li et al. CESM |ΔT|), meaning the “missing-middle” is **weaker** in more volatile climates.
  - Stored test: `thesis/body_size_stability/output_independent_stability/analysis_results.json`
  - Merged table: `thesis/body_size_stability/output_independent_stability/merged_bodymass_earthsystem.csv`

Closest prior work found so far:
- O’Gorman & Hone (2012) explicitly analyze dinosaur body-size distributions, including modality by time period and formations:
  `10.1371/journal.pone.0051925` (PDF cached at `thesis/literature/pdfs/10.1371_journal.pone.0051925.pdf`).
- Benson et al. (2014) provides the body-mass rate dataset we use: `10.1371/journal.pbio.1001853`.

What seems new (based on current review):
- Prior dinosaur body-size work discusses distribution shape (skew, sometimes multi-modal) and evolutionary drivers, but I have not
  yet found a paper that **links time-binned “missing-middle strength” to an independent deep-time climate volatility series**.

### B) Marine ecospace: climate volatility predicts functional convergence beyond taxonomy

Empirical result (stronger; still sampling-aware caveats):
- Using PBDB occurrences + PBDB ecospace traits (diet/motility/life habit) to define functional “roles”, we measure **functional
  excess similarity** (regional functional similarity after regressing out taxonomic similarity).
- When merged to independent CESM forcing, **temperature volatility** correlates positively with functional convergence even after
  controlling for time; it remains positive under additional sampling controls (collections/occurrences) and under an
  autocorrelation-aware circular-shift null.
  - Convergence bins (full PBDB): `thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv`
  - Robustness summary (sampling + circular shift; full PBDB): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb/summary.md`

Closest prior work found so far (methods + framing):
- Theoretical/ecospace framing exists (e.g., Bambach et al. 2007 theoretical ecospace): `10.1666/06054.1`.
- Many studies examine **functional/ecospace dynamics** across mass extinctions and recoveries (example: Late Triassic functional
  diversity shifts: `10.1111/pala.12332`; end-Triassic ecospace dynamics: `10.1098/rspb.2023.2232`).
- Plate tectonics/provinciality as macroevolutionary driver is well studied (e.g., “Plate tectonic regulation of global marine
  animal diversity”: `10.1073/pnas.1702297114`; classic provinciality discussion: `10.2307/3038117`).

What seems new (based on current review):
- I have not yet found work that quantifies **cross-province functional convergence** (explicitly *beyond taxonomic similarity*)
  and ties it to **independent climate-volatility forcing** over Phanerozoic time bins. Most ecospace papers focus on (a) within-
  assemblage functional diversity/composition, (b) extinction/recovery windows, or (c) plate-tectonic pacing of diversity.

New (and potentially very positionable) mechanistic signature from our own analyses:
- The volatility signal shows up as a shift in the **intercept** of `functional_similarity ~ taxonomic_similarity` (higher baseline functional similarity even among taxonomically distinct provinces), rather than a clear change in slope:
  - `thesis/synthesis/output_function_taxonomy_coupling_v3_binslopes/summary.md`

Important nuance (era heterogeneity):
- With full PBDB (0–540 Ma), the volatility→convergence association is present overall, and under within-era controls appears concentrated in the **Mesozoic** (weak in the Paleozoic in our current bins):
  - `thesis/synthesis/output_subera_volatility_convergence/summary.md`

## How publication-ready is it right now?

Promising, but still exploratory:
- The strongest potentially publishable claim is **marine functional convergence tracking independent climate volatility**, because
  it’s a broad-scale macroecological signal with an external forcing series.
- Dinosaur “missing-middle vs volatility” is intriguing but currently based on **very few bins** and needs stronger robustness.

Sampling caveat update:
- A first-pass Macrostrat rock-record covariate sensitivity check weakens the volatility→convergence association (borderline under circular shifts when adding one Macrostrat covariate; unstable if we include multiple highly-collinear Macrostrat proxies simultaneously).
  - `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_v2/summary.md`

## What would make it publication-grade (concrete next steps)

1) Sampling-aware modeling (highest priority)
- Add PBDB sampling proxies (collections count, formations, outcrop/rock area), and/or Macrostrat covariates.
- Use hierarchical models by clade/time bin (rather than simple correlations) to handle uneven sampling and autocorrelation.

2) Expand terrestrial functional convergence (to genuinely “unify” with dinosaurs)
- PBDB ecospace `jev` coverage for terrestrial taxa is thin in this pipeline; we need a better terrestrial functional dataset.
- Options: curated trait databases / ecomorph categories, or a targeted clade (e.g., tetrapods) with better trait annotation.

3) Mechanism tests for “filtering”
- The simple mediator tests (category occupancy or occupancy-heterogeneity) are not yet decisive.
- Next: look at **which roles drive pairwise similarity changes** (pair-level modeling), and whether volatility selectively
  increases the prevalence of “robust” roles *within and across* provinces in a way that predicts residual similarity.

4) Novelty hardening
- The OpenAlex list is broad and includes irrelevant hits; do a manual triage to isolate the closest 20–40 papers per claim, then
  write a crisp “what they did vs what we add” section for each.

Helpful targeted reading list (still needs triage):
- `thesis/literature/reading_lists/reading_list_coherence_openalex.md`
