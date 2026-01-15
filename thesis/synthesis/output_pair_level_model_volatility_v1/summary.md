# Pair-level model: does volatility shift functional similarity beyond taxonomy?

This is a publication-oriented upgrade of the bin-level residual correlation by fitting a **pair-level** regression and using
(i) bin-clustered robust SEs and (ii) a time-series circular-shift null on volatility.

Outcome: `functional_similarity_js` (pairwise JS similarity on role-frequency vectors).
Key predictors:
- `taxsim` = taxonomic Jaccard similarity between locality genera sets (pair-level).
- `vol_z` = standardized climate volatility (bin-level; mean |ΔT| field).
- `taxsim_x_vol_z` tests whether volatility changes the *slope* (coupling), not just the intercept.

Controls (bin-level): `time_z`, `sampling_pc1_z`, `sampling_pc2_z`, `prov_z`.

- pairs used: 27890
- bins used: 40
- sampling PCA PC1 explained variance: 0.695

## Cluster-robust inference (clusters = time bins)

- base model R²: 0.681
- vol+interaction model R²: 0.688
- vol-only model R²: 0.686

Key terms (vol+interaction):
- `vol_z` (intercept shift at taxsim=0): beta=0.0185, p_cluster=0.00176
- `taxsim_x_vol_z` (slope change): beta=-0.0763, p_cluster=0.00157

## Circular-shift null (time-series-aware; volatility shifted across bins)

- p(vol_z) exact: 0.025
- p(taxsim_x_vol_z) exact: 0.15
- p(vol_z) MC: 0.0248
- p(taxsim_x_vol_z) MC: 0.152

## Outputs

- merged pairs: `thesis/synthesis/output_pair_level_model_volatility_v1/merged_pairs.csv`
- coefficient table: `thesis/synthesis/output_pair_level_model_volatility_v1/coef_table.csv`
- stats: `thesis/synthesis/output_pair_level_model_volatility_v1/analysis_results.json`
- sampling PCA: `thesis/synthesis/output_pair_level_model_volatility_v1/sampling_pca.json`
- figures: `thesis/synthesis/output_pair_level_model_volatility_v1/figures`

Notes:
- This uses the stored pairwise sample from the convergence pipeline (not all possible pairs).
- If volatility acts mainly as a baseline shift (more similar jobs even when taxa differ), expect `vol_z > 0` and `taxsim_x_vol_z ≈ 0`.

