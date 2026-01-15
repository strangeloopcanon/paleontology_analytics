# Coherence moderation test (portfolio selectivity; exploratory)

Outcome: genus survives to the next 10 Myr bin (`survived_next`).
Key hypothesis: the survival benefit of geographic range weakens when forcing is spatially coherent.

- dataset: `thesis/geographic_portfolio/output_coherence_moderation/dataset.parquet`
- rows: 52,531
- event rate: 0.470
- grouped CV: GroupShuffleSplit by `post_bin` (held-out transitions), repeats=30
- AUC mean±sd: 0.712±0.019

## Coefficients (odds ratios per 1 SD; mean ± 95% across splits)

- log_geographic_range: OR=1.788 (95% 1.261–2.160)
- earth_delta_from_prev_T_sign_agreement_frac: OR=0.828 (95% 0.658–0.944)
- earth_delta_from_prev_T_field_meanabs: OR=1.300 (95% 1.081–1.499)
- range_x_coh_sign: OR=1.005 (95% 0.741–1.449)

Interpretation note:
- If `range_x_coh_sign` has OR < 1, range becomes *less protective* as coherence increases (supports the moderation claim).

## Outputs

- coef summary: `thesis/geographic_portfolio/output_coherence_moderation/results/coef_summary.csv`
- repeats: `thesis/geographic_portfolio/output_coherence_moderation/results/repeats.csv`
- meta: `thesis/geographic_portfolio/output_coherence_moderation/results/meta.json`
