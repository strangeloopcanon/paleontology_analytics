# Robustness battery results

## Primary specification (locked)
- Controls: time + sampling_PCA_PC12 + provinciality
- Partial correlation: 0.380
- Exact circular-shift p: 0.0500
- Resolution limit: min p = 1/40 = 0.0250

## Leave-one-out stability
- All positive: True
- Range: [0.320, 0.473]
- Mean: 0.381
- Most influential bin: 270.0 Ma

## Block bootstrap
- block_size_2: p = 0.0202
- block_size_3: p = 0.0206
- block_size_5: p = 0.0289

## Lagerstaetten exclusion
- Bins excluded: 5
- Partial correlation: 0.31869884218484185
- Exact circular-shift p: 0.08571428571428572

## SARIMAX AR order sweep
- AR(0): AIC=-160.5, BIC=-148.8, vol_beta=0.0117, vol_p=0.0789
- AR(1): AIC=-158.7, BIC=-145.4, vol_beta=0.0104, vol_p=0.162
- AR(2): AIC=-153.6, BIC=-138.9, vol_beta=0.0080, vol_p=0.325
- AR(3): AIC=-157.7, BIC=-141.5, vol_beta=0.0046, vol_p=0.399
- Best by AIC: AR(0)

## OLS + HAC (auto bandwidth)
- Auto lags: 3
- vol_beta: 0.0131
- vol_se_hac: 0.0060
- vol_p_hac: 0.0366

## Effective sample sizes
- Bins: 40
- Note: Bin-level predictors (vol_z, time_z, prov_z, sampling PCs) have effective n = n_bins. Pair-level n inflates precision for these terms unless cluster-robust SEs or mixed-effects are used.
