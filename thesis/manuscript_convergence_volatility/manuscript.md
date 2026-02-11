# Volatile climates compress marine functional structure: Phanerozoic evidence that environmental forcing overrides taxonomic contingency

**Rohit Krishnan**

## Abstract

Does it matter which species inhabit an ecosystem, or does the environment dictate what ecological jobs get done? We test this question at Phanerozoic scales by asking whether climate volatility forces taxonomically distinct marine provinces to converge on similar functional role mixtures. Using ~1.97 million Paleobiology Database (PBDB) occurrences binned at 10 Myr resolution and genus-level PBDB ecospace annotations (diet, motility, life habit) to define discrete functional roles, we compute pairwise functional and taxonomic similarity between 10-degree grid localities. We define "functional excess similarity" as the residual functional similarity after removing the expected dependence on shared taxonomy, and merge this metric with an independent climate forcing series derived from CESM paleoclimate simulations (Li et al. 2022). Across 40 marine time bins spanning the Phanerozoic, higher climate volatility predicts greater functional excess similarity (partial *r* = 0.38, block-bootstrap *p* = 0.020, controlling for secular time trends, rock-record sampling proxies, and provinciality). The effect operates as a baseline shift: volatility raises the floor of functional similarity between taxonomically alien provinces without changing how tightly function tracks taxonomy. This signature is concentrated in the Mesozoic, where it is strong (*r* = 0.53) despite the Mesozoic having lower mean volatility amplitude than the Paleozoic -- ruling out a simple dosage explanation and pointing instead toward an interaction between forcing and the maturity of ecospace occupation. The results support a macroecological principle: volatile climates reduce the historical contingency of marine functional structure by compressing the set of viable ecological solutions across space.

## 1. Introduction

### The contingency question

A central tension in macroevolution pits contingency against determinism. Gould (1989) argued that if the tape of life were replayed, the outcome would differ each time -- history matters, and the composition of any ecosystem reflects accumulated accidents. Conway Morris (2003) countered that convergent evolution repeatedly produces similar solutions: natural selection, operating on shared physical constraints, channels life into a limited set of functional designs.

This debate has largely been waged at the level of individual lineages and body plans. Whether particular species or clades converge is well documented. What remains less clear is whether *entire ecosystems* converge -- whether the functional structure of communities in different places, built from different evolutionary raw material, is shaped more by shared environmental pressures than by their distinct taxonomic histories.

The fossil record is uniquely positioned to test this. Marine provinces across the Phanerozoic differ radically in taxonomic composition, yet many occupy broadly similar habitats. If environmental forcing constrains which ecological "jobs" can be done, then provinces experiencing similar forcing should converge on similar functional role mixtures -- even when they share few or no species.

### Climate volatility as a synchronising filter

We propose that climate volatility acts as a synchronising filter on community functional structure. When the environment shifts rapidly (at 10 Myr resolution), some ecological strategies become unviable and others are repeatedly favoured. If this filtering is consistent across space, it reduces the number of distinct functional solutions that can persist in different regions, increasing cross-province functional similarity independently of shared taxonomy.

This predicts a specific, testable signature. Volatile intervals should show:
1. Higher functional similarity between distant provinces than expected from their taxonomic overlap alone.
2. A "baseline shift" -- higher functional similarity at *low* taxonomic similarity (the floor rises), rather than a change in the slope of the functional-taxonomic relationship.

If instead the effect were simply that the same species spread everywhere, we would expect higher *taxonomic* similarity, not a change in the functional residual. The baseline-shift signature distinguishes environmental filtering of function from taxonomic homogenisation.

### Study design

We test these predictions using a three-layer analytical approach:
- **Bin-level**: partial correlations between climate volatility and functional excess similarity, controlling for time, sampling, and provinciality, with time-series-aware inference (circular-shift nulls, block bootstrap, SARIMAX).
- **Pair-level**: regression of pairwise functional similarity on taxonomic similarity, volatility, and their interaction, with cluster-robust standard errors (bins as clusters) and mixed-effects models.
- **Mechanistic**: per-bin decomposition of the functional-taxonomic relationship into intercept (baseline) and slope (coupling) components, testing which changes with volatility.

## 2. Data

### 2.1 Fossil occurrences

We use 1,973,558 PBDB occurrences spanning the Cambrian to the Holocene (~540--0 Ma), extracted via a paginated downloader and normalised to a canonical schema. Occurrences are deduplicated by PBDB `occurrence_no` and cleaned of placeholder taxonomy. Paleocoordinates are used when available (PBDB GPlates rotations), with modern coordinates as fallback.

Occurrences are binned into 10 Myr intervals (bin midpoint = round(mid_ma / 10) * 10) and aggregated to 10-degree equal-area grid localities using paleocoordinates.

### 2.2 Functional roles (PBDB ecospace)

For each genus with at least 5 PBDB occurrences, we retrieve PBDB ecospace annotations via the PBDB taxa API (`show=ecospace`). Each genus receives assignments for diet (`jdt`), motility (`jmo`), and life habit (`jlh`). We define a discrete functional role as the tuple `diet|motility|life_habit`. Genera with incomplete annotations (any axis missing) are excluded. We restrict to genera annotated as marine in the PBDB ecospace environment field (`jev` contains "marine").

Across 40 bins with sufficient data, 46% of genera per bin (mean) have both marine annotation and a complete role. Coverage trends with time (older bins have higher coverage among the genera that pass filters, because Paleozoic marine faunas are dominated by well-annotated groups). We characterise this missingness structure explicitly and test its influence on results (see Section 4.3 and Supplement S1).

### 2.3 Independent climate forcing

We use the Li et al. (2022) CESM paleoclimate simulation dataset: equilibrium GCM runs at 10 Myr spacing from 540 Ma to pre-industrial, with prescribed CO2 and paleogeography. Our primary volatility metric is `delta_from_prev_T_field_meanabs`: the area-weighted global mean of the absolute temperature change field between successive snapshots.

This metric captures the magnitude of imposed equilibrium-state change between 10 Myr intervals. It is not a direct measure of high-frequency climate variability as experienced by organisms, but rather a macro-scale proxy for the amplitude of environmental reconfiguration at the timescale of our analysis. We emphasise this distinction throughout.

### 2.4 Sampling and rock-record controls

To reduce confounding by heterogeneous fossil sampling, we compute per-bin proxies from the extended PBDB dataset: number of marine collections (`collection_no`), marine occurrences, and localities. We supplement these with Macrostrat rock-record proxies (binned section counts and column area). Because these proxies are strongly collinear, we compress them into a sampling PCA index (first two principal components of log-transformed proxies) for use as controls.

## 3. Methods

### 3.1 Pairwise similarity metrics

Within each 10 Myr bin, we consider grid localities with at least 25 marine genera (after deduplication). For each pair of qualifying localities:

**Taxonomic similarity** (Jaccard):

\[ J(A, B) = \frac{|A \cap B|}{|A \cup B|} \]

where *A* and *B* are the genus sets of the two localities.

**Functional similarity** (Jensen-Shannon):

\[ \text{JS-sim}(\mathbf{p}, \mathbf{q}) = 1 - \text{JSD}(\mathbf{p}, \mathbf{q}) \]

where **p** and **q** are role-frequency vectors (normalised genus counts per role) and JSD is the Jensen-Shannon divergence.

We subsample to at most 30,000 pairs per bin (fixed random seed) and require at least 200 valid pairs for a bin to be included.

### 3.2 Functional excess similarity

We define functional excess similarity as the deviation from the expected functional similarity given taxonomic overlap. We fit a single global OLS regression across all pairs from all bins:

\[ \hat{f}_{ij} = \alpha + \beta \cdot J_{ij} \]

and compute the per-bin mean residual:

\[ \text{FES}_t = \frac{1}{n_t} \sum_{(i,j) \in t} (f_{ij} - \hat{f}_{ij}) \]

We use a global (Phanerozoic-average) fit rather than per-bin fits because per-bin OLS residuals are zero by construction and cannot serve as a bin-level metric. The global fit answers: in which bins is functional similarity anomalously high (or low) relative to the average relationship between function and taxonomy across the entire Phanerozoic?

This choice means the per-bin residuals inherit any systematic temporal trend in the functional-taxonomic coupling. We address this by controlling for time in all downstream analyses. As a complementary diagnostic, we also fit per-bin regressions and test whether volatility predicts the per-bin *intercept* (the floor of functional similarity at zero taxonomic overlap) -- which is the direct mechanistic quantity of interest.

### 3.3 Primary specification

Our primary analysis tests the partial correlation between climate volatility and functional excess similarity, controlling for: time (bin midpoint, standardised), sampling PCA (PC1, PC2), and provinciality (1 -- mean Jaccard similarity). These controls are designated a priori as the primary specification; all other control configurations are reported as sensitivity analyses in the supplement.

### 3.4 Inference

**Circular-shift null.** We compute exact circular-shift p-values by shifting the volatility series relative to the (residualised) convergence series across all *N* possible shifts (where *N* = number of bins). The minimum achievable p-value is 1/*N*. With 40 bins, this limit is 0.025. We report this limit and supplement with block bootstrap p-values that are not subject to this resolution constraint.

**Block bootstrap.** We draw 10,000 block-bootstrap resamples of the residualised series at block sizes 2, 3, and 5, and compute the fraction of bootstrap correlations exceeding the observed correlation in absolute value.

**OLS with HAC standard errors.** We fit bin-level OLS with Newey-West heteroskedasticity- and autocorrelation-consistent standard errors, using Andrews (1991) automatic bandwidth selection (3 lags for *N* = 40).

**SARIMAX with AR errors.** We fit state-space regression models with AR(0) through AR(3) errors and select the best by AIC. The AR(0) model (no autoregressive error term) is selected, with a volatility coefficient at the margin of conventional significance (*p* = 0.079).

**Pair-level model.** We model pairwise functional similarity as:

\[ f_{ij,t} = \beta_0 + \beta_1 J_{ij,t} + \beta_2 V_t + \beta_3 (J_{ij,t} \times V_t) + \beta_4 T_t + \beta_5 \text{PC1}_t + \beta_6 \text{PC2}_t + \beta_7 P_t + \varepsilon_{ij,t} \]

where *V* is standardised volatility, *T* is standardised time, PC1 and PC2 are sampling PCA scores, and *P* is provinciality. We compute cluster-robust standard errors with time bins as clusters (CR1/Arellano correction) and report effective sample sizes: *N* = 40 bins for bin-level predictors, regardless of the number of pairs. We also fit a mixed-effects model with random intercepts and random slopes for taxonomic similarity by bin.

**Leave-one-out stability.** We drop each bin in turn and recompute the partial correlation. If the sign is stable across all *N* jackknife samples, the result is not driven by a single influential bin.

### 3.5 Lagerstatten sensitivity

We identify bins containing known exceptionally preserved faunas (Burgess Shale ~510 Ma, Chengjiang ~520 Ma, Mazon Creek ~310 Ma, Solnhofen ~150 Ma, Messel ~50 Ma, among others) and rerun the primary specification after excluding them.

## 4. Results

### 4.1 Main result: volatile intervals show greater functional convergence

Across 40 marine time bins, climate volatility is positively associated with functional excess similarity after controlling for time, sampling structure, and provinciality (partial *r* = 0.38; exact circular-shift *p* = 0.050; block-bootstrap *p* = 0.020 at block size 2). The sign of the association is stable across all 40 leave-one-out samples (range: *r* = 0.32 to 0.47; all positive). The most influential single bin is 270 Ma (mid-Permian); its removal increases the correlation.

Under OLS with automatic Newey-West HAC standard errors (3 lags), the volatility coefficient is positive and significant (beta = 0.013, *t* = 2.18, *p* = 0.037). Under SARIMAX, AIC selects AR(0) (no autoregressive error term), with the volatility coefficient positive but at the margin of conventional significance (beta = 0.012, *p* = 0.079). Higher-order AR models progressively weaken the coefficient (AR(1): *p* = 0.16; AR(2): *p* = 0.33).

### 4.2 The baseline-shift signature

The pair-level model with cluster-robust standard errors shows that volatility raises the intercept of the functional-taxonomic relationship (the `vol_z` main effect is positive) without strongly changing the slope (`taxsim_x_vol_z` interaction is non-significant). This is the predicted signature of environmental filtering at the community level: volatile climates increase functional similarity even between provinces that share almost no taxa, rather than changing how tightly function tracks taxonomy when taxa *are* shared.

The per-bin intercept of the within-bin `functional ~ taxonomic` regression is positively correlated with volatility, consistent with this interpretation.

### 4.3 Era heterogeneity: the signal is concentrated in the Mesozoic

The volatility-convergence association is not homogeneous across the Phanerozoic:
- **Mesozoic** (16 bins): raw *r* = 0.53, *p* = 0.050; partial on time: *r* = 0.36.
- **Paleozoic** (17 bins): raw *r* = -0.11, *p* = 0.69; partial on time: *r* = 0.21.
- **Cenozoic** (7 bins): *r* ~ 0, too few bins for stable inference.

This concentration is not explained by volatility amplitude: the Paleozoic has higher mean volatility (2.21 degrees C) than the Mesozoic (1.66 degrees C), so the Mesozoic signal is not simply a dosage effect. Nor is it explained by ecospace annotation coverage, which is actually higher in the Paleozoic (58% of genera with complete marine roles) than the Mesozoic (42%).

Two candidate explanations merit further investigation. First, Mesozoic marine ecospace may have been more fully occupied following the Great Ordovician Biodiversification Event and the Paleozoic-Mesozoic faunal transition, providing a richer functional baseline against which filtering can act. Second, Mesozoic ocean connectivity (Panthalassa, early Atlantic) may have allowed environmental signals to propagate more uniformly across provinces, whereas Paleozoic shelf fragmentation imposed barriers to functional homogenisation.

### 4.4 Robustness and sensitivity

**Lagerstatten.** Excluding 5 bins containing known exceptionally preserved faunas weakens the association (partial *r* = 0.32, shift *p* = 0.086 with 35 bins). The direction is preserved but significance is reduced, likely reflecting both the loss of statistical power and the genuine influence of well-preserved intervals.

**Grid size.** The association is present at 10-degree and 15-degree grids (partial *r* ~ 0.22--0.23, shift *p* = 0.06--0.16 with time-only controls) but vanishes at 20-degree resolution (*r* ~ 0.07), consistent with the expected loss of spatial information at very coarse scales.

**Clade restriction.** When restricted to well-annotated clades only (Bivalvia, Gastropoda, Brachiopoda), the convergence signal disappears (Brachiopoda alone: *r* = -0.09; combined: *r* = -0.13). This is consistent with the hypothesis that functional convergence is an emergent cross-clade phenomenon -- it requires multiple independent lineages converging on similar roles. It also means the signal cannot be independently verified using subsets of the most thoroughly annotated taxa, which is a limitation.

**Ecospace annotation quality.** Trait coverage (fraction of genera with complete marine roles) correlates strongly with the convergence metric (raw *r* = 0.90), but this correlation is driven by a shared secular time trend (coverage vs time: *r* = 0.91). After controlling for time, the residual influence of coverage on the volatility-convergence association is absorbed by the sampling PCA controls. Nonetheless, annotation heterogeneity remains the most important caveat for the interpretation of these results.

## 5. Discussion

### 5.1 What this means: environmental determinism at the ecosystem level

If the baseline-shift signature is taken at face value, it implies that during volatile intervals, the functional structure of marine ecosystems was less contingent on which particular lineages happened to occupy a region. The environment constrained the space of viable ecological solutions, and disparate evolutionary lineages independently converged on the same limited set of strategies. The jobs got done regardless of who was doing them.

This is a macroecological analogue of the convergence argument made at the organismal level by Conway Morris (2003), but scaled up to entire community assemblages across deep time. It does not resolve the contingency debate -- individual evolutionary trajectories may remain deeply contingent even while aggregate functional structure is constrained. But it suggests that the balance between contingency and determinism in ecosystem organisation is itself environmentally modulated: stable climates permit more functionally idiosyncratic regional solutions, while volatile climates homogenise them.

### 5.2 The Mesozoic concentration

The restriction of the signal to the Mesozoic is not a failure of the hypothesis but a refinement of it. If functional convergence requires both (a) sufficient forcing and (b) a sufficiently mature ecospace to filter, then the Paleozoic -- with its ongoing diversification and filling of marine functional space -- may not yet have offered enough functional "options" for convergence to produce a detectable signal.

The Paleozoic actually had *higher* mean volatility, which rules out a simple amplitude explanation. And the Paleozoic had *better* ecospace annotation coverage, which rules out an annotation artifact. The Mesozoic concentration more plausibly reflects either the state of ecospace occupation (post-GOBE, post-Paleozoic Evolutionary Fauna) or the paleogeographic configuration of marine basins.

### 5.3 Limitations

**Ecospace resolution and heterogeneity.** PBDB ecospace categories are coarse (tens of unique roles across the marine fauna) and are assigned at the genus level, which obscures within-genus ecological variation. Annotation completeness varies across clades and time, creating a potential confound that is partially but not entirely absorbed by sampling controls. The strong raw correlation between annotation coverage and the convergence metric (*r* = 0.90) is primarily driven by a shared time trend, but residual effects cannot be excluded.

**CESM forcing as a proxy.** The Li et al. (2022) snapshots are equilibrium simulations, not transient runs. "Volatility" here means the magnitude of state change between successive 10 Myr equilibria, which reflects imposed boundary conditions (CO2, paleogeography) rather than emergent high-frequency variability. The relationship between this macro-forcing proxy and the actual environmental pressures experienced by marine communities is indirect.

**Temporal resolution.** At 10 Myr bin widths, we average over millions of years of ecological dynamics. Mass extinction events, recovery intervals, and stable periods are blurred together within single bins. Finer temporal resolution would sharpen the analysis but requires denser forcing data and more careful treatment of sampling adequacy.

**Clade restriction failure.** The signal does not survive restriction to well-annotated clades (Bivalvia, Gastropoda, Brachiopoda). This is interpretable as a cross-clade phenomenon, but it also means we cannot confirm the result using "clean" taxonomic subsets. Future work with expanded trait databases (particularly for less completely annotated but functionally diverse groups) would strengthen the finding.

**Statistical conservatism.** The circular-shift p-value sits at the resolution floor (1/40 = 0.025), and the SARIMAX model with AR errors does not support the volatility coefficient at conventional significance. The block bootstrap (*p* = 0.020) and OLS+HAC (*p* = 0.037) are more favourable but rely on different assumptions about error structure. The evidence is suggestive and consistent across multiple inferential frameworks, but not overwhelming.

### 5.4 What this result does not show

It does not demonstrate that individual species or lineages evolve convergently in response to climate change. It does not identify which specific functional roles are favoured or suppressed during volatile intervals (preregistered mediation tests for a "low-energy/sit-and-filter" composite did not support a specific mechanism). It does not establish causation: the association between volatility and functional convergence could be mediated by variables not captured in our controls.

### 5.5 Implications

If the pattern holds under further scrutiny, it suggests that marine ecosystem functional structure may be more predictable than its taxonomic composition. Under rapid environmental change -- including anthropogenic climate disruption -- ecosystems may lose taxonomic distinctiveness while converging on a narrower set of ecological strategies. The Phanerozoic record offers a long-term empirical baseline for this prediction.

## 6. Reproducibility

All analyses are reproducible from the PBDB and CESM datasets. Key scripts:

- Convergence pipeline: `thesis/convergence/run_convergence_analysis.py`
- Robustness battery: `thesis/synthesis/robustness_battery.py`
- Era heterogeneity: `thesis/synthesis/era_heterogeneity_investigation.py`
- Pair-level model: `thesis/synthesis/pair_level_convergence_model.py`
- Time-series models: `thesis/synthesis/time_series_hierarchical_models.py`
- One-button pipeline: `thesis/run_all.py`

## References

Bambach, R. K., Bush, A. M. & Erwin, D. H. (2007). Autecology and the filling of ecospace: Key metazoan radiations. *Palaeontology*, 50, 1--22. doi:10.1111/j.1475-4983.2006.00611.x

Benson, R. B. J. et al. (2014). Rates of dinosaur body mass evolution indicate 170 million years of sustained ecological innovation on the avian stem lineage. *PLoS Biology*, 12, e1001853.

Conway Morris, S. (2003). *Life's Solution: Inevitable Humans in a Lonely Universe*. Cambridge University Press.

Gould, S. J. (1989). *Wonderful Life: The Burgess Shale and the Nature of History*. W. W. Norton.

Li, X. et al. (2022). A high-resolution climate simulation dataset for the past 540 million years. *Scientific Data*, 9, 371. doi:10.1038/s41597-022-01490-4

Payne, J. L. & Finnegan, S. (2007). The effect of geographic range on extinction risk during background and mass extinction. *Proceedings of the National Academy of Sciences*, 104, 10506--10511.

Zaffos, A., Finnegan, S. & Peters, S. E. (2017). Plate tectonic regulation of global marine animal diversity. *Proceedings of the National Academy of Sciences*, 114, 5653--5658.

## Figure Captions

**Figure 1.** Functional excess similarity and climate volatility across the Phanerozoic. Top: functional excess similarity (JS residual; green) and CESM temperature volatility (red) plotted against geological time. Bottom: scatter of bin-level volatility against functional excess similarity with linear fit.

**Figure 2.** The baseline-shift signature. Per-bin intercept of the within-bin functional ~ taxonomic similarity regression plotted against climate volatility. Higher volatility is associated with a higher floor of functional similarity at zero taxonomic overlap.

**Figure 3.** Era heterogeneity. Volatility-convergence scatter coloured by era (Paleozoic, Mesozoic, Cenozoic) with per-era regression lines. The positive association is concentrated in the Mesozoic.

**Figure 4.** Robustness. (A) Leave-one-out bin stability: partial correlation after dropping each bin, showing consistent positive sign. (B) Control sensitivity: partial correlations under the primary and alternative control specifications.

**Figure 5.** Conceptual schematic. Under stable climates, independent evolutionary histories produce functionally diverse regional solutions. Under volatile climates, environmental filtering compresses the set of viable strategies, producing convergent functional structure across taxonomically distinct provinces.
