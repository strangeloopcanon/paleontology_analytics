"""Shared statistical and utility helpers for thesis analysis scripts.

Canonical implementations drawn from robustness_battery.py, run_pipeline.py,
and test_independent_stability.py. Import from here instead of copy-pasting.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def z_score(x: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    """Z-score an array, returning NaN for positions that were NaN in input."""
    x = x.astype(float)
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if int(np.sum(mask)) < 3:
        return out
    mu = float(np.mean(x[mask]))
    sd = float(np.std(x[mask], ddof=ddof))
    sd = sd if sd > 0 else 1.0
    out[mask] = (x[mask] - mu) / sd
    return out


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals of y on X (with intercept). NaN-safe."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    if int(np.sum(mask)) < X.shape[1] + 3:
        return out
    A = np.column_stack([np.ones(int(np.sum(mask))), X[mask]])
    beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
    out[mask] = y[mask] - A.dot(beta)
    return out


def partial_corr(v: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Partial Pearson correlation of v and y after residualizing on controls."""
    rv = residualize(v, controls)
    ry = residualize(y, controls)
    mask = np.isfinite(rv) & np.isfinite(ry)
    if int(np.sum(mask)) < 6:
        return float("nan")
    return float(np.corrcoef(rv[mask], ry[mask])[0, 1])


def pca_scores(X: np.ndarray, *, k: int) -> np.ndarray:
    """First k PCA scores via SVD. NaN-safe row masking."""
    mask = np.all(np.isfinite(X), axis=1)
    if int(np.sum(mask)) < max(6, k + 3):
        return np.full((len(X), k), np.nan)
    Xc = X[mask]
    mu, sd = np.mean(Xc, 0), np.std(Xc, 0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    U, S, Vt = np.linalg.svd((Xc - mu) / sd, full_matrices=False)
    kk = min(k, Vt.shape[0])
    scores = np.full((len(X), k), np.nan)
    scores[mask, :kk] = U[:, :kk] * S[:kk]
    return scores


def build_controls(bins: pd.DataFrame) -> np.ndarray:
    """Primary specification controls: time + sampling_PCA_PC12 + provinciality."""
    feat_cols = []
    for col in ["n_localities", "marine_n_collections", "marine_n_occurrences"]:
        if col in bins.columns:
            feat_cols.append(np.log1p(bins[col].to_numpy(dtype=float)))
    for col in ["macro_col_area_sum", "macro_n_sections"]:
        if col in bins.columns:
            feat_cols.append(np.log1p(bins[col].to_numpy(dtype=float)))
    t = bins["time_bin"].to_numpy(dtype=float)
    prov = bins["provinciality"].to_numpy(dtype=float) if "provinciality" in bins.columns else np.zeros(len(bins))
    if feat_cols:
        pcs = pca_scores(np.column_stack(feat_cols), k=2)
        return np.column_stack([t, pcs[:, 0], pcs[:, 1], prov])
    return np.column_stack([t, prov])


def circular_shift_p(v: np.ndarray, y: np.ndarray, controls: np.ndarray) -> dict:
    """Exact circular-shift p-value for partial correlation."""
    rv = residualize(v, controls)
    ry = residualize(y, controls)
    mask = np.isfinite(rv) & np.isfinite(ry)
    rv, ry = rv[mask], ry[mask]
    n = len(rv)
    if n < 6:
        return {"corr": float("nan"), "p_shift": float("nan"), "n": n}
    obs = float(np.corrcoef(rv, ry)[0, 1])
    more = sum(1 for s in range(n) if abs(float(np.corrcoef(rv, np.roll(ry, s))[0, 1])) >= abs(obs))
    return {"corr": obs, "p_shift": more / n, "n": n}


def perm_test_corr(
    x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int
) -> dict[str, float]:
    """Two-sided permutation test for Pearson r. NaN-safe."""
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6:
        return {"corr": float("nan"), "p_perm": float("nan"), "n": float(len(x))}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(int(permutations)):
        yp = rng.permutation(y)
        c = float(np.corrcoef(x, yp)[0, 1])
        if abs(c) >= abs(obs):
            more_extreme += 1
    p = (more_extreme + 1) / (int(permutations) + 1)
    return {"corr": obs, "p_perm": float(p), "n": float(len(x))}


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two points (WGS-84 mean radius)."""
    r = 6371.0088
    lat1, lng1, lat2, lng2 = (
        math.radians(v) for v in (lat1, lng1, lat2, lng2)
    )
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def circular_mean_deg(longitudes_deg: np.ndarray) -> float:
    """Directional (circular) mean of angles in degrees."""
    rad = np.deg2rad(longitudes_deg)
    s = float(np.nanmean(np.sin(rad)))
    c = float(np.nanmean(np.cos(rad)))
    return float(np.rad2deg(np.arctan2(s, c)))
