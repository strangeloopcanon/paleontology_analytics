from __future__ import annotations

import pandas as pd


_PLACEHOLDER_TOKENS = frozenset({"nan", "none", "null"})


def clean_taxon_series(series: pd.Series) -> pd.Series:
    """
    Normalize a taxonomy-like string column.

    - Preserves missing values as <NA>
    - Trims whitespace
    - Converts placeholder strings like "nan"/"None" into <NA>
    """
    s = series.astype("string").str.strip()
    s = s.mask(s == "", pd.NA)
    lowered = s.str.lower()
    return s.mask(lowered.isin(_PLACEHOLDER_TOKENS), pd.NA)


def clean_taxonomy_label(series: pd.Series, *, unclassified_label: str = "Unclassified") -> pd.Series:
    """
    Normalize higher-level taxonomy fields (phylum/class/order/family).

    In addition to `clean_taxon_series`, this collapses PBDB placeholders like
    "NO_CLASS_SPECIFIED" into a single `unclassified_label`.
    """
    s = clean_taxon_series(series)
    s = s.mask(s.str.match(r"^NO_.*_SPECIFIED$", na=False), pd.NA)
    return s.fillna(unclassified_label)

