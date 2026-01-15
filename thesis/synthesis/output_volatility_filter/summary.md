# Synthesis test: volatility-as-filter (marine convergence) + alignment with dinosaur barbell metric

Marine ecospace test (PBDB ecospace roles, locality grid + 10 Myr bins):
- volatility proxy: CESM |ΔT| field mean absolute change (Li et al. 2022)
- convergence: excess similarity of full role composition (JS residual vs taxonomic similarity)
- mediator: filter index from a priori guild occupancy sets (robust minus specialized strategies)

## Core results (controls: time + log1p(n_localities))

- Total: volatility → convergence: corr=0.241, perm-p=0.106, n=46
- Path a: volatility → filter index: corr=0.500, perm-p=0.00055, n=46
- Path b: filter → convergence (controls volatility): corr=-0.123, perm-p=0.419, n=46
- Direct: volatility → convergence (controls filter): corr=0.268, perm-p=0.0721, n=46

## Mediation (standardized OLS; bootstrap 95% CI)

- n=46, a=0.542, b=0.049, direct=0.050, indirect=0.027
- indirect 95% CI: [-0.063, 0.140]

## Mesozoic slice (70–200 Ma)

- volatility → convergence: corr=-0.030, perm-p=0.917, n=14
- volatility → filter index: corr=-0.112, perm-p=0.698, n=14
- mediation (n=14): indirect=0.017, 95% CI=(-0.2803760403725651, 0.34415466770273684)

## Dinosaur alignment check (Avialae-included, mass2; small n)

- volatility → gap_ratio_hist: corr=0.853, perm-p=0.0101, n=8
- marine convergence ↔ dinosaur gap_ratio_hist: corr=0.497, perm-p=0.201, n=8
- partial corr (controls time+volatility): corr=0.107, perm-p=0.812, n=8

## Files

- Marine merged table: `thesis/synthesis/output_volatility_filter/merged_marine_filter_convergence.csv`
- Mesozoic marine+dino: `thesis/synthesis/output_volatility_filter/merged_mesozoic_marine_dino.csv`
- Stats: `thesis/synthesis/output_volatility_filter/analysis_results.json`
- Figures: `thesis/synthesis/output_volatility_filter/figures`

## Interpretation guardrails

- Time bins are autocorrelated; permutation/bootstrap here treat bins as exchangeable (use as hypothesis test, not final inference).
- The filter index uses coarse ecospace categories and should be stress-tested with alternative sets and sampling controls (PBDB collections, Macrostrat rock area).

