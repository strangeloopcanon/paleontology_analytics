from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import netcdf_file
from scipy.ndimage import label


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _area_weights(lat_deg: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg.astype(float))
    w = np.cos(lat_rad)
    w = np.clip(w, 0.0, None)
    return w / float(np.sum(w)) if float(np.sum(w)) > 0 else w


def _weighted_mean(field_lat_lon: np.ndarray, w_lat: np.ndarray) -> float:
    # field shape: (lat, lon); weights: (lat,)
    x = field_lat_lon.astype(float)
    if x.ndim != 2:
        raise ValueError("Expected (lat,lon) field")
    w = w_lat.reshape(-1, 1)
    denom = float(np.sum(w)) * x.shape[1]
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * w) / denom)


def _weighted_mean_abs_diff(a: np.ndarray, b: np.ndarray, w_lat: np.ndarray) -> float:
    return _weighted_mean(np.abs(a.astype(float) - b.astype(float)), w_lat)


def _grid_cell_weights(w_lat: np.ndarray, n_lon: int) -> np.ndarray:
    # Returns weights with shape (lat, lon) that sum to 1 across the grid.
    w = w_lat.astype(float).reshape(-1, 1)
    denom = float(n_lon)
    if denom <= 0:
        return np.zeros((len(w_lat), max(n_lon, 1)), dtype=float)
    return np.repeat(w / denom, repeats=int(n_lon), axis=1)


def _weighted_fraction(mask_true: np.ndarray, mask_valid: np.ndarray, w_cell: np.ndarray) -> float:
    mt = mask_true.astype(bool)
    mv = mask_valid.astype(bool)
    if mt.shape != w_cell.shape or mv.shape != w_cell.shape:
        raise ValueError("Masks must match weight shape")
    denom = float(np.sum(w_cell[mv]))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w_cell[mv & mt]) / denom)


def _sign_edge_count(sign_map: np.ndarray) -> int:
    # Count adjacent edges with opposite sign (ignore zeros) with longitude wrap.
    s = sign_map.astype(np.int8)
    east = np.roll(s, -1, axis=1)
    h = int(np.sum((s != east) & (s != 0) & (east != 0)))
    north = s[1:, :]
    v = int(np.sum((s[:-1, :] != north) & (s[:-1, :] != 0) & (north != 0)))
    return h + v


def _morans_i(field: np.ndarray) -> float:
    # Simple 4-neighbor Moran's I on a regular grid with longitude wrap; unweighted.
    x = field.astype(float)
    if x.ndim != 2:
        raise ValueError("Expected (lat,lon) field")
    if not np.all(np.isfinite(x)):
        x = x.copy()
        x[~np.isfinite(x)] = np.nan

    mean = float(np.nanmean(x))
    xm = x - mean
    denom = float(np.nansum(xm**2))
    if denom <= 0:
        return float("nan")

    east = np.roll(xm, -1, axis=1)
    num_east = float(np.nansum(xm * east))
    num_north = float(np.nansum(xm[:-1, :] * xm[1:, :]))
    num = num_east + num_north

    n_lat, n_lon = x.shape
    n = float(n_lat * n_lon)
    w = float((n_lat * n_lon) + ((n_lat - 1) * n_lon))  # east + north edges
    if w <= 0 or n <= 0:
        return float("nan")
    return float((n / w) * (num / denom))


def _svd_coherence_metrics(field: np.ndarray, w_lat: np.ndarray) -> dict[str, float]:
    # Treat a single (lat,lon) field as a matrix; measure low-dimensional dominance.
    x = field.astype(float)
    if x.ndim != 2:
        raise ValueError("Expected (lat,lon) field")
    w = np.sqrt(w_lat.astype(float).reshape(-1, 1))
    xw = x * w
    s = np.linalg.svd(xw, full_matrices=False, compute_uv=False)
    s2 = s**2
    total = float(np.sum(s2))
    if total <= 0:
        return {"pc1_frac": float("nan"), "effective_rank": float("nan"), "participation_ratio": float("nan")}

    p = s2 / total
    pc1 = float(p[0]) if len(p) else float("nan")
    pp = p[p > 0]
    ent = float(-np.sum(pp * np.log(pp))) if len(pp) else float("nan")
    eff_rank = float(np.exp(ent)) if np.isfinite(ent) else float("nan")
    part = float(1.0 / float(np.sum(p**2))) if float(np.sum(p**2)) > 0 else float("nan")
    return {"pc1_frac": pc1, "effective_rank": eff_rank, "participation_ratio": part}


def _coastline_index(mask: np.ndarray) -> int:
    # mask shape: (lat, lon)
    mask = mask.astype(bool)
    # longitude wrap
    h = int(np.sum(mask != np.roll(mask, -1, axis=1)))
    # latitude no wrap
    v = int(np.sum(mask[:-1, :] != mask[1:, :]))
    return h + v


def _components_count(mask: np.ndarray) -> int:
    mask = mask.astype(bool)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    _, n = label(mask, structure=structure)
    return int(n)


def _components_count_with_lon_wrap_heuristic(mask: np.ndarray) -> int:
    # Avoid “dateline splitting” by taking the minimum component count across two longitude origins.
    n0 = _components_count(mask)
    n1 = _components_count(np.roll(mask, mask.shape[1] // 2, axis=1))
    return int(min(n0, n1))


def _plot_series(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.plot(d[x], d[y], marker="o", linewidth=1.4, color="#1f77b4")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if x == "time_ma":
        ax.invert_xaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nc", default="data/raw/external/climate_540myr/High_Resolution_Climate_Simulation_Dataset_540_Myr.nc")
    p.add_argument("--out", default="thesis/earth_system/climate_540myr/output")
    p.add_argument("--land-threshold", type=float, default=0.5)
    p.add_argument("--dt-cell-eps-c", type=float, default=0.1, help="Cell-level |ΔT| threshold for sign-based coherence metrics (°C).")
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    nc_path = Path(args.nc)
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing NetCDF: {nc_path}")

    rows: list[dict[str, Any]] = []
    with netcdf_file(nc_path, "r", mmap=True) as nc:
        lat = nc.variables["lat"][:].copy()
        w_lat = _area_weights(lat)

        T = nc.variables["T"]
        P = nc.variables["P"]
        LANDFRAC = nc.variables["LANDFRAC"]

        n_sim = int(T.shape[0])
        if n_sim != 55:
            print(f"Warning: expected 55 simulations, got {n_sim}")

        prev_T_mean_field = None
        prev_land = None
        prev_coast = None
        prev_components = None

        for i in range(n_sim):
            time_ma = float(540 - 10 * i)

            # Monthly mean fields.
            T_mean_field = T[i, :, :, :].mean(axis=0).astype(np.float32)
            P_mean_field = P[i, :, :, :].mean(axis=0).astype(np.float32)
            land = LANDFRAC[i, :, :].astype(np.float32)

            T_global = _weighted_mean(T_mean_field, w_lat)
            P_global = _weighted_mean(P_mean_field, w_lat)
            land_global = _weighted_mean(land, w_lat)

            land_mask = land > float(args.land_threshold)
            coast = _coastline_index(land_mask)
            components = _components_count_with_lon_wrap_heuristic(land_mask)

            # Deltas relative to previous (older) snapshot.
            if prev_T_mean_field is None:
                dT_global_abs = float("nan")
                dT_field_meanabs = float("nan")
                dT_coherence_ratio = float("nan")
                dT_sign_agree = float("nan")
                dT_sign_edges = float("nan")
                dT_sign_components = float("nan")
                dT_morans_i = float("nan")
                dT_pc1_frac = float("nan")
                dT_effective_rank = float("nan")
                dT_participation_ratio = float("nan")
                dP_global_abs = float("nan")
                dLand_field_meanabs = float("nan")
                dCoast_abs = float("nan")
                dComponents_abs = float("nan")
            else:
                dT_signed = float(T_global - prev_T_global)  # type: ignore[name-defined]
                dT_global_abs = float(abs(T_global - prev_T_global))  # type: ignore[name-defined]
                dP_global_abs = float(abs(P_global - prev_P_global))  # type: ignore[name-defined]
                dT_field_meanabs = _weighted_mean_abs_diff(T_mean_field, prev_T_mean_field, w_lat)

                # Spatial coherence of ΔT: "is the world changing together?"
                dT = (T_mean_field.astype(np.float32) - prev_T_mean_field.astype(np.float32)).astype(np.float32)
                denom = float(dT_field_meanabs) if np.isfinite(dT_field_meanabs) else float("nan")
                dT_coherence_ratio = float(abs(dT_signed) / denom) if denom and denom > 0 else float("nan")

                w_cell = _grid_cell_weights(w_lat, n_lon=int(dT.shape[1]))
                cell_eps = float(args.dt_cell_eps_c)
                valid = np.isfinite(dT) & (np.abs(dT) >= cell_eps)
                global_sign = 0.0 if not np.isfinite(dT_signed) else float(np.sign(dT_signed))
                if global_sign == 0.0:
                    dT_sign_agree = float("nan")
                else:
                    agree = np.isfinite(dT) & (np.sign(dT) == global_sign)
                    dT_sign_agree = _weighted_fraction(agree, valid, w_cell)

                sign_map = np.zeros_like(dT, dtype=np.int8)
                sign_map[dT > cell_eps] = 1
                sign_map[dT < -cell_eps] = -1
                dT_sign_edges = float(_sign_edge_count(sign_map))
                dT_sign_components = float(
                    _components_count_with_lon_wrap_heuristic(sign_map == 1) + _components_count_with_lon_wrap_heuristic(sign_map == -1)
                )
                dT_morans_i = float(_morans_i(dT))
                svd = _svd_coherence_metrics(dT, w_lat)
                dT_pc1_frac = float(svd["pc1_frac"])
                dT_effective_rank = float(svd["effective_rank"])
                dT_participation_ratio = float(svd["participation_ratio"])

                dLand_field_meanabs = _weighted_mean_abs_diff(land, prev_land, w_lat) if prev_land is not None else float("nan")
                dCoast_abs = float(abs(coast - int(prev_coast))) if prev_coast is not None else float("nan")
                dComponents_abs = float(abs(components - int(prev_components))) if prev_components is not None else float("nan")

            rows.append(
                {
                    "time_ma": time_ma,
                    "T_global_mean_c": float(T_global),
                    "P_global_mean_mm_month": float(P_global),
                    "land_area_fraction": float(land_global),
                    "land_components": int(components),
                    "coastline_index": int(coast),
                    "delta_from_prev_T_global_abs": float(dT_global_abs),
                    "delta_from_prev_T_field_meanabs": float(dT_field_meanabs),
                    "delta_from_prev_T_coherence_ratio": float(dT_coherence_ratio),
                    "delta_from_prev_T_sign_agreement_frac": float(dT_sign_agree),
                    "delta_from_prev_T_sign_edge_count": float(dT_sign_edges),
                    "delta_from_prev_T_sign_components": float(dT_sign_components),
                    "delta_from_prev_T_morans_i": float(dT_morans_i),
                    "delta_from_prev_T_pc1_frac": float(dT_pc1_frac),
                    "delta_from_prev_T_effective_rank": float(dT_effective_rank),
                    "delta_from_prev_T_participation_ratio": float(dT_participation_ratio),
                    "delta_from_prev_P_global_abs": float(dP_global_abs),
                    "delta_from_prev_landfrac_field_meanabs": float(dLand_field_meanabs),
                    "delta_from_prev_coastline_abs": float(dCoast_abs),
                    "delta_from_prev_land_components_abs": float(dComponents_abs),
                }
            )

            prev_T_mean_field = T_mean_field
            prev_land = land
            prev_coast = coast
            prev_components = components
            prev_T_global = T_global
            prev_P_global = P_global

    df = pd.DataFrame(rows).sort_values("time_ma", ascending=False).reset_index(drop=True)
    out_csv = out_dir / "climate_540myr_timeseries.csv"
    df.to_csv(out_csv, index=False)

    # Figures for quick inspection.
    _plot_series(df, x="time_ma", y="T_global_mean_c", out_path=fig_dir / "T_global_mean_c.png", title="Global mean surface temperature (°C)")
    _plot_series(
        df,
        x="time_ma",
        y="delta_from_prev_T_global_abs",
        out_path=fig_dir / "delta_T_global_abs.png",
        title="Climate volatility proxy: |Δ global mean temperature| (10 Myr steps)",
    )
    _plot_series(
        df,
        x="time_ma",
        y="delta_from_prev_landfrac_field_meanabs",
        out_path=fig_dir / "delta_landfrac_meanabs.png",
        title="Paleogeography change proxy: mean |Δ LANDFRAC| (10 Myr steps)",
    )
    _plot_series(
        df,
        x="time_ma",
        y="coastline_index",
        out_path=fig_dir / "coastline_index.png",
        title="Coastline index (grid-edge land/sea transitions)",
    )

    # Summary markdown.
    lines = [
        "# Derived time series: climate + paleogeography (Li et al. 2022 CESM snapshots)",
        "",
        f"- NetCDF: `{nc_path}`",
        f"- Output CSV: `{out_csv}`",
        "",
        "## Notes",
        "",
        "- Simulations are ordered as in the authors’ `extract_data.ncl`: 540 Ma, 530 Ma, ..., 10 Ma, PI (0 Ma).",
        "- `LANDFRAC` is used as a paleogeography proxy (land–sea distribution).",
        "- Volatility metrics are “from previous snapshot” differences (10 Myr step).",
        "",
        "## Variables",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `T_global_mean_c` | area-weighted global mean monthly-mean surface temperature |",
        "| `P_global_mean_mm_month` | area-weighted global mean monthly-mean precipitation |",
        "| `land_area_fraction` | area-weighted mean land fraction |",
        "| `land_components` | approximate number of land components (`LANDFRAC > threshold`) |",
        "| `coastline_index` | grid-edge land/sea transition count (proxy for coastline complexity) |",
        "| `delta_from_prev_*` | absolute change from previous (older) snapshot |",
        "| `delta_from_prev_T_coherence_ratio` | coherence proxy: `|Δ global mean T| / mean(|ΔT field|)` (≈1 means mostly same-sign change globally) |",
        "| `delta_from_prev_T_sign_agreement_frac` | coherence proxy: fraction of cells whose ΔT sign matches the global mean ΔT sign |",
        "| `delta_from_prev_T_sign_edge_count` | patchiness proxy: number of adjacent grid edges where ΔT sign flips |",
        "| `delta_from_prev_T_sign_components` | patchiness proxy: number of connected components in warming + cooling sign masks |",
        "| `delta_from_prev_T_morans_i` | patchiness proxy: Moran’s I of ΔT field (4-neighbor) |",
        "| `delta_from_prev_T_pc1_frac` | coherence proxy: rank-1 dominance of ΔT field (SVD pc1 energy fraction) |",
        "| `delta_from_prev_T_effective_rank` | coherence proxy: effective rank of ΔT field (lower = more low-dimensional) |",
        "| `delta_from_prev_T_participation_ratio` | coherence proxy: participation ratio of ΔT field singular spectrum |",
        "",
        "## Figures",
        "",
        f"- `{fig_dir}`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
