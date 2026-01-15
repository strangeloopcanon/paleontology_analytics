# Synthesis test: volatility-as-filter (marine convergence) + alignment with dinosaur barbell metric

Marine ecospace test (PBDB ecospace roles, locality grid + 10 Myr bins):
- volatility proxy: CESM |ΔT| field mean absolute change (Li et al. 2022)
- convergence: excess similarity of full role composition (JS residual vs taxonomic similarity)
- mediator: filter index from a priori guild occupancy sets (robust minus specialized strategies)

## Core results (controls: time + log1p(n_localities))

- Total: volatility → convergence: corr=0.163, perm-p=0.277, n=46
- Path a: volatility → filter index: corr=0.530, perm-p=0.00035, n=46
- Path b: filter → convergence (controls volatility): corr=0.092, perm-p=0.547, n=46
- Direct: volatility → convergence (controls filter): corr=0.092, perm-p=0.549, n=46

## Mediation (standardized OLS; bootstrap 95% CI)

- n=46, a=0.542, b=0.049, direct=0.050, indirect=0.027
- indirect 95% CI: [-0.063, 0.140]

## Alternative mediator: homogenization (−mean p(1−p) across prevalent categories)

- Path a: volatility → homogenization: corr=0.279, perm-p=0.0589, n=46
- Path b: homogenization → convergence (controls volatility): corr=0.255, perm-p=0.0842, n=46
- Direct: volatility → convergence (controls homogenization): corr=0.090, perm-p=0.547, n=46

- mediation (homogenization): n=46, a=0.189, b=0.183, direct=0.043, indirect=0.035
- indirect 95% CI: [-0.010, 0.120]

## Robustness: original convergence metric (PBDB ecospace v2 output)

- Total: volatility → convergence_v2: corr=0.351, perm-p=0.0449, n=33
- Path a: volatility → homogenization: corr=0.334, perm-p=0.0601, n=33
- Path b: homogenization → convergence_v2 (controls volatility): corr=0.206, perm-p=0.251, n=33
- Direct: volatility → convergence_v2 (controls homogenization): corr=0.279, perm-p=0.114, n=33
- mediation via homogenization: n=33, indirect=0.034, 95% CI=(-0.027825037054889444, 0.1621608617033108)

### Mesozoic slice (70–200 Ma; v2 metric)

- volatility → convergence_v2: corr=-0.113, perm-p=0.708, n=13
- volatility → homogenization: corr=0.145, perm-p=0.635, n=13

## Mesozoic slice (70–200 Ma)

- volatility → convergence: corr=0.049, perm-p=0.865, n=14
- volatility → filter index: corr=-0.113, perm-p=0.698, n=14
- volatility → homogenization: corr=0.364, perm-p=0.209, n=14
- mediation (n=14): indirect=0.017, 95% CI=(-0.2803760403725651, 0.34415466770273684)
- mediation via homogenization (n=14): indirect=-0.138, 95% CI=(-0.6041288545473493, 0.3103129611621984)

## Dinosaur alignment check (Avialae-included, mass2; small n)

- volatility → gap_ratio_hist: corr=0.853, perm-p=0.0101, n=8
- marine convergence ↔ dinosaur gap_ratio_hist: corr=0.497, perm-p=0.201, n=8
- partial corr (controls time+volatility): corr=0.276, perm-p=0.497, n=8

## Files

- Marine merged table: `thesis/synthesis/output_volatility_filter_v4/merged_marine_filter_convergence.csv`
- Mesozoic marine+dino: `thesis/synthesis/output_volatility_filter_v4/merged_mesozoic_marine_dino.csv`
- Stats: `thesis/synthesis/output_volatility_filter_v4/analysis_results.json`
- Figures: `thesis/synthesis/output_volatility_filter_v4/figures`

## Interpretation guardrails

- Time bins are autocorrelated; permutation/bootstrap here treat bins as exchangeable (use as hypothesis test, not final inference).
- The filter index uses coarse ecospace categories and should be stress-tested with alternative sets and sampling controls (PBDB collections, Macrostrat rock area).

