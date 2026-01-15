# Functional↔taxonomic coupling under forcing (first pass)

Each bin fits `functional_similarity ~ taxonomic_similarity` across province pairs; we test whether the fitted slope changes with forcing.

Controls: time + sampling proxies (localities/collections/occurrences) + provinciality; iid + circular-shift p-values.

## Results

- js_slope__vs_vol: corr=-0.084, iid_p=0.605, shift_p=0.511, n=40
- js_slope__vs_coh_sign: corr=0.026, iid_p=0.87, shift_p=0.873, n=40
- js_slope__vs_eff_rank: corr=0.242, iid_p=0.136, shift_p=0.179, n=40

- roles_slope__vs_vol: corr=0.018, iid_p=0.911, shift_p=0.848, n=40
- roles_slope__vs_coh_sign: corr=0.098, iid_p=0.552, shift_p=0.584, n=40
- roles_slope__vs_eff_rank: corr=0.023, iid_p=0.888, shift_p=0.863, n=40

- js_intercept__vs_vol: corr=0.327, iid_p=0.0382, shift_p=0.0244, n=40
- js_intercept__vs_coh_sign: corr=0.310, iid_p=0.0506, shift_p=0.125, n=40
- js_intercept__vs_eff_rank: corr=-0.385, iid_p=0.0132, shift_p=0.0236, n=40

- roles_intercept__vs_vol: corr=0.051, iid_p=0.758, shift_p=0.723, n=40
- roles_intercept__vs_coh_sign: corr=0.045, iid_p=0.781, shift_p=0.766, n=40
- roles_intercept__vs_eff_rank: corr=-0.134, iid_p=0.416, shift_p=0.329, n=40

Interpretation:
- Negative corr(slope, forcing) implies functional similarity becomes less dependent on taxonomic similarity (more decoupling).
- Positive corr(slope, forcing) implies tighter coupling (functions track taxa more).
- Positive corr(intercept, forcing) implies higher baseline functional similarity at a given (low) taxonomic similarity (a decoupling signature).

## Outputs

- merged: `thesis/synthesis/output_function_taxonomy_coupling_v3_binslopes/merged.csv`
- results: `thesis/synthesis/output_function_taxonomy_coupling_v3_binslopes/analysis_results.json`
