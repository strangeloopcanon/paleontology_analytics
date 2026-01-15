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

Interpretation:
- Negative corr(slope, forcing) implies functional similarity becomes less dependent on taxonomic similarity (more decoupling).
- Positive corr(slope, forcing) implies tighter coupling (functions track taxa more).

## Outputs

- merged: `thesis/synthesis/output_function_taxonomy_coupling_v2_binslopes/merged.csv`
- results: `thesis/synthesis/output_function_taxonomy_coupling_v2_binslopes/analysis_results.json`
