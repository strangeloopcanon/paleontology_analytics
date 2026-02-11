# Ecospace missingness diagnostic

- Bins analysed: 54
- Mean fraction with complete role (all genera): 0.764
- Mean fraction marine + complete role: 0.462

## Temporal trend in annotation quality

- corr_frac_has_role_vs_convergence: corr=0.852, p=0.0001, n=40
- corr_frac_marine_with_role_vs_convergence: corr=0.900, p=0.0001, n=40
- corr_frac_in_ecospace_vs_convergence: corr=0.785, p=0.0001, n=40
- corr_frac_has_role_vs_time: corr=0.925, p=0.0001, n=40
- corr_frac_marine_with_role_vs_time: corr=0.910, p=0.0001, n=40

## Interpretation

If `frac_marine_with_role` correlates strongly with `functional_excess_similarity_js`,
then annotation completeness may confound the convergence metric. If the correlation is
weak or non-significant, the signal is unlikely to be driven by trait missingness alone.

## Files

- Coverage table: `thesis/synthesis/output_ecospace_missingness/ecospace_coverage_per_bin.csv`
- Merged table: `thesis/synthesis/output_ecospace_missingness/merged_coverage_convergence.csv`
- Stats: `thesis/synthesis/output_ecospace_missingness/analysis_results.json`
- Figures: `thesis/synthesis/output_ecospace_missingness/figures`
