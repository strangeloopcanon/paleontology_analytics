# Literature review (structured, DOI-anchored)

This review is intentionally organized around *methods* and *failure modes* that directly impact an occurrence-derived mobility project.

## A. Climate velocity: definition, computation, and pitfalls

- **Foundational definition**: Loarie et al. (2009) introduced “climate velocity” as the ratio of temporal climate change to spatial gradients (`10.1038/nature08649`).
- **Algorithmic sensitivity**: Hamann et al. (2014) compared velocity algorithms and conservation use cases (`10.1111/gcb.12736`), emphasizing that computation choices (e.g., gradient estimation) affect conclusions.
- **Known bias/underestimation regimes**: Dobrowski & Parks (2016) showed underestimation in mountainous regions (`10.1038/ncomms12349`), illustrating that complex gradients break naïve velocity calculations.

## B. Biotic velocity: observed community/taxon responses vs climatic forcing

- **Late Quaternary linkage to endemism**: Sandel et al. (2011) connect climate-change velocity to endemism patterns (`10.1126/science.1210173`).
- **Explicit separation of climatic vs biotic velocities**: Ordóñez & Williams (2013) estimated both velocities for woody taxa over the last 16 kyr (`10.1111/ele.12110`), providing a conceptual template for “mobility from fossil occurrences” and warning that biotic and climatic velocities can diverge.
- **Biotic vs climatic vulnerability mapping**: Carroll et al. (2015) formalized biotic velocity as a vulnerability metric (`10.1371/journal.pone.0140486`).
- **Past↔future range shifts and dispersal**: Williams & Blois (2018) discuss how dispersal capability interacts with climate velocities (`10.1111/jbi.13395`).

## C. Marine taxa tracking climate velocity: strong effects and mismatches

- **Empirical tracking**: Pinsky et al. (2013) show marine taxa often track local climate velocities (`10.1126/science.1239352`).
- **Forecast redistribution**: García Molinos et al. (2015) develop climate velocity redistribution projections (`10.1038/nclimate2769`).
- **Trait mediation**: Sunday et al. (2015) link traits + velocity to range shifts (`10.1111/ele.12474`).
- **Mismatch evidence**: Chivers et al. (2017) document mismatches between plankton movements and climate velocity (`10.1038/ncomms14434`), cautioning against assuming tracking.

## D. Deep-time spatial data infrastructure: PBDB, paleogeography, plate models

- **PBDB programmatic access**: Peters & McClennen (2015) document the PBDB API (`10.1017/pab.2015.39`); Varela et al. (2014) provide the `paleobioDB` R package (`10.1111/ecog.01154`).
- **PBDB operational documentation**: Uhen et al. (2023) is a citable PBDB user guide (`10.5070/p9401160531`); PBDB’s dataset handle exists via GBIF (`10.15468/bmsbpj`).
- **Plate reconstructions**: GPlates is a central tool for plate kinematics (`10.1029/2018gc007584`). Cao et al. (2017) show paleobiology can improve paleogeography reconstructions (`10.5194/bg-14-5425-2017`), reinforcing that uncertainty is two-way (paleodata ↔ plate models).
- **Tectonics and biodiversity**: Zaffos et al. (2017) integrate plate tectonics with PBDB diversity (`10.1073/pnas.1702297114`); Leprieur et al. (2016) is another tectonics-driven biodiversity study (`10.1038/ncomms11461`).
- **Coordinate transformations/paleolatitude**: van Hinsbergen et al. (2015) provide paleolatitude tools and caveats (`10.1371/journal.pone.0126946`).

## E. Extinction selectivity: established predictors and model structure

This work’s mobility proxy sits inside a mature extinction-selectivity literature where range/breadth/age are repeatedly implicated:

- **Geographic range as buffer**: Payne & Finnegan (2007) show range size reduces extinction risk across regimes (`10.1073/pnas.0701257104`).
- **Multivariate extinction risk**: Harnik (2011) decomposes drivers in fossil bivalves (`10.1073/pnas.1100572108`).
- **Environmental breadth and longevity**: Heim & Peters (2011) link environmental breadth → range/longevity (`10.1371/journal.pone.0018946`).
- **Niche breadth + range size**: Saupe et al. (2015) test survival determinants at geological time scales (`10.1111/geb.12333`).
- **Rarity frameworks**: Harnik et al. (2012) synthesize “forms of rarity” and extinction risk (`10.1098/rspb.2012.1902`).
- **Age selectivity**: Finnegan et al. (2008) revisit age selectivity (“Red Queen”) in marine genera (`10.1666/07008.1`).
- **Habitat breadth and dynamics**: Nürnberg & Aberhan (2013) connect habitat breadth and geographic range to diversity dynamics (`10.1666/12047`).
- **Event-specific selectivity**: Finnegan et al. (2012) show climatic forcing and selectivity at the Late Ordovician (`10.1073/pnas.1117039109`); Crampton et al. (2016) show regime shifts in extinction (`10.1073/pnas.1519092113`).

## F. Fossil sampling, spatial standardization, and “disappearance” vs extinction

Any mobility-from-occurrences project must treat sampling as a first-class problem:

- **Sampling standardization**: Alroy’s SQS and related work emphasize fair sampling (`10.1017/S1089332600001819`), with tooling in `divDyn` (Kocsis et al. 2019 `10.1111/2041-210X.13161`).
- **Spatial standardization emphasis**: Antell et al. (2024) call for spatial standardization of occurrence data (`10.1017/pab.2023.36`)—directly relevant for centroid-based displacement metrics.
- **Sampling-aware turnover inference**: CMR methods (Connolly & Miller 2001 `10.1666/0094-8373(2001)027<0751:JEOSAT>2.0.CO;2`; Liow & Nichols 2010 `10.1017/S1089332600001820`) and Bayesian preservation models (Silvestro et al. 2014 `10.1093/sysbio/syu006`; PyRate `10.1111/2041-210X.12263`) are the methodological gold standard for separating sampling from true extinction/origination.
- **Coordinate cleaning practice (adjacent best practice)**: `CoordinateCleaner` (Zizka et al. 2019 `10.1111/2041-210X.13152`) is not fossil-specific, but illustrates standardized coordinate QC expectations in occurrence pipelines.

## G. Takeaway for this project

The literature supports mobility as a plausible mechanism and “velocity” metrics as informative descriptors, but also makes it clear that:

1) **Velocity metrics are method-sensitive** (computation choices matter).
2) **Observed biotic movement often mismatches climatic forcing**.
3) **Deep-time occurrence data require explicit spatial/sampling controls** to avoid confounding.

Accordingly, the contribution must be defined as much by its *bias controls and null models* as by the mobility metric itself.

## H. Geographic range as an extinction buffer (what is established)

The strongest and most repeatedly supported geographic predictor of survivorship in deep time is **geographic range size** (often proxied by occupancy / number of localities).

- **Range-size selectivity across regimes**: Payne & Finnegan (2007) explicitly tested background vs mass-extinction regimes and found geographic range size reduces extinction risk (`10.1073/pnas.0701257104`).
- **Spatial dynamics framing**: Jablonski (2008) emphasizes extinction as a spatial process and highlights how geographic structure shapes persistence (`10.1073/pnas.0801919105`).
- **Trait–environment interactions over Phanerozoic scales**: Clade/trait mediation can modulate geographic buffering (e.g., “trait–environment interactions” framing; `10.1111/gcb.12963`).

## I. Range configuration beyond size: fragmentation, connectedness, and “geographic portfolios”

Range size is not the whole story: two taxa with the same occupied-area proxy can differ in *configuration* (compact vs disjunct; single-core vs multi-core; latitudinally narrow vs broad). Configuration connects naturally to survivorship mechanisms:

- **Fragmentation/connectivity theory**: Connectivity can reduce local extinction via recolonization (“rescue effect”), while fragmentation can increase risk via isolation; metapopulation work formalizes these mechanisms (`10.1890/0012-9658(2002)083[3243:cfaeri]2.0.co;2`, `10.1111/cobi.12047`, `10.1046/j.1523-1739.1999.013002314.x`).
- **Empirical deep-time selectivity context**: Event-specific and clade-specific selectivity can differ, and selectivity can weaken/shift during the largest crises (`10.1098/rsos.230795`), motivating explicit tests of *which* geographic dimensions remain protective when the kill mechanism changes.
- **Configuration can be measurable from occurrences**: even with coarse fossil data, grid-occupancy networks provide computable summaries (number of components, dominance of a “core” component, latitudinal spread) that are distinct from pure range size if controlled for sampling intensity.

## J. Empirical anchors linking geography to extinction selectivity (adjacent lines of evidence)

These references motivate why it is reasonable to test configuration metrics at scale, and how to interpret differences across events:

- **Event-level geographic selectivity**: studies emphasize that extinction risk drivers can vary across events and clades; for example, Late Ordovician selectivity shows a climate-linked signature (`10.1073/pnas.1117039109`).
- **Distribution and extinction risk around crises**: Triassic–Jurassic work explicitly ties geographic distribution to extinction risk in marine benthos (`10.1111/j.1365-2699.2007.01709.x`).
- **Abundance and extinction risk in late Paleozoic/early Mesozoic clades**: abundance-at-multiple-scales is associated with extinction risk (`10.1666/10037.1`), highlighting the need to separate sampling intensity from geography itself.
- **Network/structure perspectives**: network-style approaches can reveal selectivity gradients (e.g., end-Cretaceous bivalve “network” selectivity; `10.1038/srep01790`) even when classic predictors are known.

## K. Methods: estimating configuration from occurrence data (and why sampling is still the hard part)

For any occurrence-derived geography metric, the main methodological question is whether the metric is stable to sampling irregularities.

- **Sampling completeness as a confound**: sampling can bias perceived geography and extinction risk; sampling-aware evaluations in the fossil record explicitly quantify this (`10.1017/pab.2019.43`) and should be treated as required context for any new geography metric.
- **Preservation of range-size distributions**: range-size signals can be surprisingly well preserved under some conditions (`10.1086/710176`), which motivates testing whether *configuration* metrics also retain interpretable structure.
- **Practical computation choices**:
  - Grid occupancy (e.g., 5° bins) vs convex-hull methods trade bias/variance differently.
  - Connected-component metrics depend on neighborhood definitions and longitude wrap; sensitivity checks are necessary.
  - Rarefaction / spatial subsampling can reduce dependence on “number of occurrences” and help interpretability.

## L. Takeaway for a strong contribution (beyond “method novelty”)

If mobility proxies fail negative controls, a strong alternative biological question is:

> **Do taxa with different *geographic portfolio structures* (connected-core vs fragmented multi-core; narrow vs broad latitudinal spread) show different survivorship across major crises, after controlling for range size and sampling intensity?**

This reframes the contribution from “a new mobility metric” to a **new, testable, mechanism-linked hypothesis about how geographic structure mediates extinction vulnerability**, and it is directly interpretable in terms of rescue effects, spatial heterogeneity of kill mechanisms, and dispersal corridors.

## Appendix: broader reading list (OpenAlex keyword searches)

- Auto-built list (titles/venues/DOIs, sorted by citations): `thesis/literature/reading_lists/portfolio_extinction_reading_list.md`
- The raw OpenAlex exports (`*.json`) are intentionally treated as local cache (large + easy to regenerate).
