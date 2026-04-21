# Preregistration: Climate Volatility and Functional Convergence

Locked before running confirmatory experiments (Angle A / Angle B).
Commit-tagged at registration time.

---

## Primary hypothesis

When climate changes rapidly between consecutive 10 Myr intervals, geographically
distant marine fossil assemblages converge on more similar ecological role mixtures
— even when they share few or no genera.

## Outcome variable

**Functional excess similarity (JS)** — `functional_excess_similarity_js`

Computed as follows:

1. For each 10 Myr time bin, assign PBDB marine occurrences to 10° paleolatitude ×
   paleolongitude grid cells ("localities").
2. For each locality in a bin, build a role-frequency vector: the number of unique
   genera in each ecological role, where `role = diet | motility | life_habit`
   (PBDB ecospace fields `jdt`, `jmo`, `jlh`; all three required, environment must
   contain "marine").
3. Compute pairwise Jensen–Shannon similarity (`1 − JS_distance`, base-2) between
   role-frequency vectors for all locality pairs within a bin.
4. Compute pairwise taxonomic similarity (Jaccard on genus sets) for the same pairs.
5. Fit a single global OLS: `functional_similarity_js ~ taxonomic_similarity` across
   all pairs from all bins. The residual for each pair is "functional excess similarity."
6. Average the residuals within each time bin to produce one `functional_excess_similarity_js`
   value per bin (the bin-level outcome).

**Defaults:** `min_genera_per_region = 25`, `max_pairs_per_bin = 30000`,
`min_occ_per_genus = 5`, `grid_deg = 10`, `time_bin_myr = 10`.

## Exposure variable

**Climate volatility** — `delta_from_prev_T_field_meanabs`

Area-weighted (cosine latitude) spatial mean of absolute cell-wise temperature change
between consecutive CESM Li et al. (2022) snapshot simulations, spaced at 10 Myr
intervals across the Phanerozoic (540–0 Ma). Derived from monthly-mean annual-cycle
temperature fields averaged per snapshot.

Source script: `thesis/earth_system/climate_540myr/derive_timeseries.py`.

## Control set (primary specification)

**Time + sampling PCA (PC1, PC2) + provinciality** — matching `_build_controls()` in
`thesis/synthesis/robustness_battery.py`.

| Variable | Construction |
|----------|-------------|
| `time_bin` | Mid-point age of bin (Ma) |
| `sampling_pc1`, `sampling_pc2` | First two PCA scores of log1p-transformed sampling features: `n_localities`, `marine_n_collections`, `marine_n_occurrences` (plus `macro_col_area_sum`, `macro_n_sections` if available from Macrostrat) |
| `provinciality` | `1 − mean pairwise Jaccard (genus sets)` across localities within a bin |

**Coverage-controlled specification (co-primary):** the above plus `frac_marine_with_role`
(fraction of marine genera in a bin that have all three ecospace fields annotated).

## Primary inference tests (bin-level)

Both must be reported; the claim stands if **either** clears its threshold.

| Test | Specification | Decision threshold |
|------|--------------|-------------------|
| **Joint circular block bootstrap** | Block sizes {2, 3, 5}; 10,000 replicates; joint resampling of `(v_resid, y_resid)` blocks; two-sided `\|r\| ≥ \|r_obs\|` | p < 0.05 for at least one block size |
| **SARIMAX AR(1)** | Endogenous = `functional_excess_similarity_js`; exogenous = `[vol_z, time_z, pc1_z, pc2_z, prov_z]`; `order=(1,0,0)`, `trend="c"` | p < 0.10 for volatility coefficient |

**Secondary tests (reported but not decision-relevant):**
- Exact circular-shift p (all N shifts)
- Random circular-shift p (20,000 draws)
- HAC OLS (Newey–West, Andrews-rule lags)
- Leave-one-out bin stability (range of partial r across all single-bin drops)
- SARIMAX AR(0), AR(2), AR(3) sweeps

## Pair-level specification (mechanism — baseline shift)

**Outcome:** `functional_similarity_js` (raw, not excess)

**Regression:**
`func_sim_js = β₀ + β₁·taxsim + β₂·vol_z + β₃·(taxsim × vol_z) + β₄·time_z + β₅·pc1_z + β₆·pc2_z + β₇·prov_z + ε`

- Cluster-robust standard errors (CR1 / Arellano), clusters = `time_bin`.
- `vol_z` = z-scored `delta_from_prev_T_field_meanabs` at the bin level.
- `taxsim` = pairwise Jaccard (genus sets).

**Baseline-shift claim:** β₂ > 0 (volatility raises functional similarity floor)
and β₃ ≈ 0 (does not change the taxonomic–functional coupling slope).

**Confirmatory non-parametric test (new):** quantile regression of `func_sim_js` on
`vol_z` at the 10th, 25th, 50th, and 75th percentiles of `taxsim`, plus stratified
comparison of mean `func_sim_js` for pairs with `taxsim < 0.1` across volatility
tertiles, with bin-cluster bootstrap CIs. The mechanism claim is supported if the
low-taxsim quantile shows a positive `vol_z` effect.

## Confirmatory experiments (Angle A)

These experiments are run after preregistration commit. Results reported regardless
of outcome.

| Experiment | Description | Decision |
|-----------|-------------|----------|
| **Spatial null** | Within-bin locality-label permutation (10,000 reps). Preserves marginal genus/role distributions per cell; breaks pair geography. | Claim survives if observed partial-r exceeds 95th percentile of null |
| **Joint block bootstrap** | Replace y-only bootstrap with joint (v,y) block resample | Reported alongside existing y-only for transparency |
| **Grid sensitivity (full controls)** | Rerun at 10°, 15°, 20° with _build_controls() (time + PC1,2 + prov), not time-only | Signal survival at ≥ 2 of 3 grid sizes supports robustness |
| **Coverage confound** | (i) Errors-in-variables / Deming regression on `frac_marine_with_role`, (ii) restrict to top-quartile annotated genera, (iii) decompose coverage into linear + quadratic + detrended components | Headline survives if partial-r remains positive and p < 0.05 under any one attack |
| **Exposure portfolio** | Headline partial-r under 5 exposures: field |ΔT|, global |ΔT|, |ΔP|, max-cell |ΔT|, land-area change | Climate story supported if ≥ 2 climate exposures give r > 0.25 and paleogeography gives |r| < 0.15 |

## Exploratory experiments (Angle B)

Results are hypothesis-generating; framed as exploratory in the manuscript.

| Experiment | Purpose |
|-----------|---------|
| Mesozoic mechanism mediation | Test whether role-saturation, post-Permian vacancy, or MMR metrics mediate the era effect |
| Functional fingerprint | Identify which ecological roles consistently expand under volatile climates |
| Clade decomposition | Which clade pairs contribute most to JS convergence in high-vol bins |
| Forward prediction | Train on 540–270 Ma, predict 270–0 Ma out-of-sample |
| Modern analog | Identify deep-time bins matching anthropogenic-scale |ΔT| |

## Analysis defaults

| Parameter | Value |
|-----------|-------|
| `time_bin_myr` | 10 |
| `grid_deg` | 10 |
| `min_genera_per_region` | 25 |
| `max_pairs_per_bin` | 30,000 |
| `min_occ_per_genus` | 5 |
| `seed` | 42 (convergence), 77 (synthesis) |
| `n_permutations` | 20,000 (circular shift), 10,000 (bootstrap) |
| Block sizes (bootstrap) | {2, 3, 5} |

## Data sources

- **PBDB:** Paleobiology Database bulk download (all Phanerozoic marine occurrences
  with ecospace fields). Accessed via `src/acquisition/pbdb.py`.
- **CESM:** Li et al. (2022) paleoclimate simulations, 55 snapshots at 10 Myr resolution.
  Processed via `thesis/earth_system/climate_540myr/derive_timeseries.py`.
- **Macrostrat:** Collection-area and section counts per bin (optional sampling controls).

## Genuine residual caveats

These are structural and will not be resolved by the experiments above:

- **n ≈ 40 time bins** for the bin-level time series. This is inherent to
  Phanerozoic-scale work at 10 Myr resolution.
- **Single CESM trajectory.** No ensemble spread; the volatility metric reflects one
  model's reconstruction of one climate history. This is the current state of the art
  in deep-time paleoclimate modeling.
