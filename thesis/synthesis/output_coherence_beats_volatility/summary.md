# Coherence beats magnitude? (first pass, full PBDB)

Outcome: `functional_excess_similarity_js` (marine functional convergence beyond taxonomy).
Magnitude: `delta_from_prev_T_field_meanabs` (mean |ΔT| field, 10 Myr step).
Coherence: `delta_from_prev_T_coherence_ratio` = `|Δ global mean T| / mean(|ΔT field|)` (≈1 means mostly same-sign changes).

- bins used: 40
- corr(vol, coh_ratio): 0.725
- R2 base (time+sampling+prov): 0.898
- R2 + volatility: 0.911
- R2 + coherence_ratio: 0.909
- R2 + both: 0.912

## Partial correlation tests (controls noted in name; iid + circular-shift p-values)

- coh_ratio__control_time: corr=0.321, iid_p=0.0408, shift_p=0.105, n=40
- coh_ratio__control_time_loc: corr=0.282, iid_p=0.0767, shift_p=0.127, n=40
- coh_ratio__control_time_loc_samp: corr=0.303, iid_p=0.0561, shift_p=0.18, n=40
- coh_ratio__control_time_loc_samp_prov: corr=0.333, iid_p=0.0349, shift_p=0.128, n=40

- coh_ratio__control_time_loc_samp_prov_plus_vol: corr=0.097, iid_p=0.555, shift_p=0.592, n=40
- vol__control_time_loc_samp_prov_plus_coh_ratio: corr=0.169, iid_p=0.294, shift_p=0.255, n=40

## Additional coherence/patchiness metrics (sanity checks)

- coh_sign__control_time_loc_samp_prov: corr=0.386, iid_p=0.0134, shift_p=0.0541, n=40
- patch_edges__control_time_loc_samp_prov: corr=-0.316, iid_p=0.0508, shift_p=0.0743, n=40
- pc1_frac__control_time_loc_samp_prov: corr=0.335, iid_p=0.0345, shift_p=0.0536, n=40
- effective_rank__control_time_loc_samp_prov: corr=-0.378, iid_p=0.0156, shift_p=0.0262, n=40

## Outputs

- merged: `thesis/synthesis/output_coherence_beats_volatility/merged.csv`
- results: `thesis/synthesis/output_coherence_beats_volatility/analysis_results.json`
- figures: `thesis/synthesis/output_coherence_beats_volatility/figures`

Notes:
- These are bin-level tests; publication-grade inference should use explicit time-series or hierarchical models.
- Coherence metrics are derived from global ΔT fields (Li et al. 2022 CESM snapshots) and are sensitive to the chosen ΔT threshold.
