# New hypothesis candidate: volatility increases “role interchangeability” (taxa swap, roles persist)

## Core claim (mechanistic; testable)

When climate volatility is higher, ecological roles become **less taxon-specific**: many different taxa can occupy the same small set
of viable roles. This produces functional convergence even when taxonomic similarity stays low.

Operational prediction:
- A time bin’s climate volatility predicts **lower taxon↔role association strength**, even after controlling for sampling and time.

## Why this is different from what we already tested

Our current headline result is about **between-province functional similarity beyond taxonomy** (a spatial pattern). This hypothesis
targets a different object:

- the **mapping** between taxa and roles (how interchangeable taxa are with respect to roles), not just whether provinces look similar.

## One concrete metric family

Compute, per time bin (globally or by province), a statistic like:

- Mutual information `I(Taxon; Role)` (or normalized MI) between genus (or family) and role (`diet|motility|life_habit`),
  estimated from occurrence counts or locality-unique counts.

Interpretation:
- High MI: roles are clade-specific (knowing the role narrows the likely taxa).
- Low MI: roles are taxonomically interchangeable (knowing the role tells you little about taxa).

Prediction:
- Volatility ↑ ⇒ MI ↓.

## What would falsify it

- MI does not decrease with volatility once sampling proxies (collections/occurrences/rock proxies) and time-series dependence are
  controlled.
- MI trends are fully explained by a single clade takeover (i.e., taxonomic composition changes, not interchangeability).

## Why it might matter (“so what”)

If true, volatility doesn’t just “select against specialists”; it changes the *organization* of biodiversity:
it makes ecosystems more **replaceable** in terms of “who does which jobs”, implying that extinction risk and recovery can be
decoupled from taxonomic loss if functional roles remain occupied.

## Prior-art check (status)

I cannot guarantee “no prior art”. A quick OpenAlex keyword scan did not surface a clear deep-time paper using MI on PBDB ecospace,
but the scan is broad and needs manual triage:

- `thesis/literature/reading_lists/reading_list_taxon_role_information_openalex.md`
