# Supplement (draft): climate volatility and marine functional convergence

## S1. File map (outputs)

Core results:
- Pair-level model (cluster robust + exact circular-shift p-values): `thesis/synthesis/output_pair_level_model_volatility_v1/summary.md`
- Sampling + autocorr robustness (bin-level): `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb/summary.md`
- Macrostrat + sampling PCA robustness: `thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/summary.md`
- Functional↔taxonomic coupling signature: `thesis/synthesis/output_function_taxonomy_coupling_v3_binslopes/summary.md`

Interpretability + mechanisms:
- Role/job shifts with volatility (occupancy + mean shares; many-tests caveat): `thesis/synthesis/output_role_jobs_volatility_v1/summary.md`
- Low-energy composite mediation attempt (negative): `thesis/synthesis/output_low_energy_index_mediation_v1/summary.md`

Alternative inference:
- Time-series + hierarchical fits: `thesis/synthesis/output_time_series_hierarchical_models_v1/summary.md`

## S2. Statistical details (pair-level model)

Outcome:
- Pairwise functional similarity between locality role-frequency vectors (Jensen–Shannon similarity).

Key predictors:
- `taxsim`: Jaccard similarity of genus sets.
- `vol_z`: standardized volatility (mean |ΔT| field between successive CESM snapshots).
- `taxsim_x_vol_z`: interaction, testing whether volatility changes the functional↔taxonomic coupling slope.

Controls (bin-level):
- `time_z`
- sampling PCA (`pc1_z`, `pc2_z`) computed from PBDB sampling proxies + Macrostrat
- `prov_z` (provinciality = 1 - mean taxonomic similarity)

Inference:
- OLS with CR1 (Arellano) bin-clustered robust SEs.
- Time-series-aware p-values via **exact circular shifts** of `vol_z` across bins (minimum attainable exact p = 1/n_bins).

## S3. Statistical details (bin-level time-series models)

Outcome:
- Bin-level `functional_excess_similarity_js` (average residual functional similarity after removing dependence on taxonomic similarity).

Models:
- OLS with iid SEs.
- OLS with Newey–West HAC SEs (lag=1).
- SARIMAX AR(1) residual process.

Motivation:
- Make serial dependence explicit and quantify how sensitive the volatility coefficient is to AR error assumptions.

## S4. Low-energy composite index definition

We predefine a single composite index to avoid a many-comparisons story:

- low-energy diets: {suspension feeder, deposit feeder, detritivore}
- high-energy diets: {carnivore, piscivore}
- low-energy motility: {stationary, slow-moving, passively mobile}
- high-energy motility: {actively mobile, fast-moving}
- low-energy life habits: {epifaunal, infaunal, semi-infaunal}
- high-energy life habits: {nektonic, nektobenthic, aquatic, aquatic, depth=surface}

Index per bin:
`index_raw = (diet_low - diet_high) + (mot_low - mot_high) + (hab_low - hab_high)`, then standardized to `index_z`.

Result:
- Index does not track volatility under circular shifts and does not attenuate the volatility coefficient materially.

Interpretation:
- Convergence is unlikely to be explained purely by a global mean shift toward low-energy composition (as defined here).

## S5. Reproducibility

`python thesis/run_all.py`

