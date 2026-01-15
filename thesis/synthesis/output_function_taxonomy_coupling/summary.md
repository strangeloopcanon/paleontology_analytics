# Functional↔taxonomic coupling under forcing (first pass)

Each bin fits `functional_similarity ~ taxonomic_similarity` across province pairs; we test whether the fitted slope changes with forcing.

Controls: time + sampling proxies (localities/collections/occurrences) + provinciality; iid + circular-shift p-values.

## Results

- js_slope__vs_vol: corr=-0.000, iid_p=0.999, shift_p=1, n=40
- js_slope__vs_coh_sign: corr=0.000, iid_p=1, shift_p=1, n=40
- js_slope__vs_eff_rank: corr=0.000, iid_p=0.998, shift_p=1, n=40

- roles_slope__vs_vol: corr=0.000, iid_p=1, shift_p=1, n=40
- roles_slope__vs_coh_sign: corr=0.000, iid_p=1, shift_p=1, n=40
- roles_slope__vs_eff_rank: corr=-0.000, iid_p=1, shift_p=1, n=40

Interpretation:
- Negative corr(slope, forcing) implies functional similarity becomes less dependent on taxonomic similarity (more decoupling).
- Positive corr(slope, forcing) implies tighter coupling (functions track taxa more).

## Outputs

- merged: `thesis/synthesis/output_function_taxonomy_coupling/merged.csv`
- results: `thesis/synthesis/output_function_taxonomy_coupling/analysis_results.json`
