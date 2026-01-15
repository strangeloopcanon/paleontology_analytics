# Time-series + hierarchical inference (publication-grade upgrade)

We fit two complementary models:
1) **Bin-level time-series regression** on `functional_excess_similarity_js` with AR errors (SARIMAX) and HAC SE sensitivity.
2) **Pair-level mixed effects** model with random bin effects (and, when possible, a random taxsim slope).

## Bin-level (n bins)
- bins: 40

Volatility coefficient (`vol_z`) across bin-level models:
- OLS: beta=0.0130, se=0.0054, p=0.0221
- OLS + HAC: beta=0.0130, se=0.0057, p=0.029
- SARIMAX AR(1): beta=0.0102, se=0.0073, p=0.162

## Pair-level mixed effects
- bins: 40, pairs: 27890
- model: MixedLM_re_int+taxsim
- vol_z: beta=0.0167, se=0.0064, p=0.00891

## Outputs
- bin table: `thesis/synthesis/output_time_series_hierarchical_models_v1/bins_model_table.csv`
- bin coefficients: `thesis/synthesis/output_time_series_hierarchical_models_v1/bin_model_coefs.csv`
- pair mixedLM coefficients: `thesis/synthesis/output_time_series_hierarchical_models_v1/pair_mixedlm_coefs.csv`
- meta: `thesis/synthesis/output_time_series_hierarchical_models_v1/meta.json`
- figures: `thesis/synthesis/output_time_series_hierarchical_models_v1/figures`
