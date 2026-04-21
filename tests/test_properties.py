"""Property-based tests for core numeric utilities.

Uses inline implementations so that the test suite is self-contained and does
not depend on thesis/_lib/ existing.
"""
from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings, strategies as st
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Inline implementations under test
# ---------------------------------------------------------------------------

def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres between two points on Earth."""
    r = 6_371.0
    lat1, lng1, lat2, lng2 = (math.radians(v) for v in (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def z_score(arr: np.ndarray) -> np.ndarray:
    """Standardise *arr* to zero mean and unit variance."""
    std = arr.std(ddof=0)
    if std == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - arr.mean()) / std


def pca_scores(x: np.ndarray, k: int) -> np.ndarray:
    """Return the first *k* PCA scores for *x* (n x m)."""
    pca = PCA(n_components=k)
    return pca.fit_transform(x)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

latitudes = st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False)
longitudes = st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# 1. Haversine symmetry
# ---------------------------------------------------------------------------

@given(lat1=latitudes, lng1=longitudes, lat2=latitudes, lng2=longitudes)
def test_haversine_symmetry(lat1: float, lng1: float, lat2: float, lng2: float) -> None:
    assert math.isclose(haversine(lat1, lng1, lat2, lng2), haversine(lat2, lng2, lat1, lng1), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 2. Haversine non-negativity
# ---------------------------------------------------------------------------

@given(lat1=latitudes, lng1=longitudes, lat2=latitudes, lng2=longitudes)
def test_haversine_non_negative(lat1: float, lng1: float, lat2: float, lng2: float) -> None:
    assert haversine(lat1, lng1, lat2, lng2) >= 0.0


# ---------------------------------------------------------------------------
# 3. Haversine triangle inequality
# ---------------------------------------------------------------------------

@given(
    lat1=latitudes, lng1=longitudes,
    lat2=latitudes, lng2=longitudes,
    lat3=latitudes, lng3=longitudes,
)
def test_haversine_triangle_inequality(
    lat1: float, lng1: float,
    lat2: float, lng2: float,
    lat3: float, lng3: float,
) -> None:
    ac = haversine(lat1, lng1, lat3, lng3)
    ab = haversine(lat1, lng1, lat2, lng2)
    bc = haversine(lat2, lng2, lat3, lng3)
    assert ac <= ab + bc + 1e-6  # small tolerance for FP rounding


# ---------------------------------------------------------------------------
# 4. z_score produces mean ≈ 0 and std ≈ 1
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=200,
    )
)
def test_z_score_mean_zero_std_one(values: list[float]) -> None:
    arr = np.array(values)
    if arr.std(ddof=0) == 0:
        return  # constant arrays are trivially zero-filled
    z = z_score(arr)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std(ddof=0) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 5. PCA output shape
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=3, max_value=30),
    m=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=20, deadline=None)
def test_pca_scores_shape(n: int, m: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, m))
    k = rng.integers(1, min(n, m) + 1)
    scores = pca_scores(x, int(k))
    assert scores.shape == (n, k)
