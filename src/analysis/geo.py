import pandas as pd


def add_analysis_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `analysis_lat` / `analysis_lng` columns for downstream analyses.

    Preference order:
    - Use paleocoordinates (`paleolat` / `paleolng`) when present for an occurrence.
    - Fall back to modern coordinates (`lat` / `lng`) otherwise.
    """
    if "analysis_lat" in df.columns and "analysis_lng" in df.columns:
        return df

    paleolat = df["paleolat"] if "paleolat" in df.columns else None
    paleolng = df["paleolng"] if "paleolng" in df.columns else None
    lat = df["lat"] if "lat" in df.columns else None
    lng = df["lng"] if "lng" in df.columns else None

    if paleolat is not None and lat is not None:
        df["analysis_lat"] = paleolat.where(paleolat.notna(), lat)
    elif lat is not None:
        df["analysis_lat"] = lat
    else:
        df["analysis_lat"] = pd.NA

    if paleolng is not None and lng is not None:
        df["analysis_lng"] = paleolng.where(paleolng.notna(), lng)
    elif lng is not None:
        df["analysis_lng"] = lng
    else:
        df["analysis_lng"] = pd.NA

    return df


def add_binned_locality(
    df: pd.DataFrame,
    *,
    lat_col: str = "analysis_lat",
    lng_col: str = "analysis_lng",
    bin_degrees: float = 5.0,
    locality_col: str = "locality",
) -> pd.DataFrame:
    """
    Add a discrete locality identifier based on binned lat/lng.
    """
    if lat_col not in df.columns or lng_col not in df.columns:
        raise KeyError(f"Missing required columns: {lat_col}, {lng_col}")

    df[f"{lat_col}_bin"] = (df[lat_col] / bin_degrees).round() * bin_degrees
    df[f"{lng_col}_bin"] = (df[lng_col] / bin_degrees).round() * bin_degrees
    df[locality_col] = list(zip(df[f"{lat_col}_bin"], df[f"{lng_col}_bin"]))
    return df

