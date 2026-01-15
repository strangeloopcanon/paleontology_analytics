from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt


PLOS_DATASET_S1_URL = "https://journals.plos.org/plosbiology/article/file"
PLOS_DATASET_S1_PARAMS = {"type": "supplementary", "id": "10.1371/journal.pbio.1001853.s011"}
PBDB_INTERVALS_URL = "https://paleobiodb.org/data1.2/intervals/list.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _circular_mean_deg(longitudes_deg: np.ndarray) -> float:
    rad = np.deg2rad(longitudes_deg.astype(float))
    s = np.nanmean(np.sin(rad))
    c = np.nanmean(np.cos(rad))
    return float(np.rad2deg(np.arctan2(s, c)))


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0088
    lat1 = np.deg2rad(lat1)
    lng1 = np.deg2rad(lng1)
    lat2 = np.deg2rad(lat2)
    lng2 = np.deg2rad(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def _moment_bimodality(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if sd == 0:
        return float("nan")
    z = (x - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    if kurt == 0:
        return float("nan")
    return float((skew**2 + 1.0) / kurt)


def _gap_ratio_hist(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 15:
        return float("nan")
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr <= 0:
        return float("nan")
    h = 2.0 * iqr / (n ** (1.0 / 3.0))
    if not np.isfinite(h) or h <= 0:
        return float("nan")

    n_bins = int(np.clip(math.ceil((float(np.max(x)) - float(np.min(x))) / h), 10, 50))
    counts, edges = np.histogram(x, bins=n_bins)
    if counts.sum() == 0:
        return float("nan")
    dens = counts.astype(float) / float(counts.sum())

    # Simple smoothing.
    if len(dens) >= 5:
        dens = np.convolve(dens, np.array([0.25, 0.5, 0.25]), mode="same")

    centers = (edges[:-1] + edges[1:]) / 2
    med = float(np.median(x))
    med_idx = int(np.argmin(np.abs(centers - med)))

    maxima = []
    for i in range(1, len(dens) - 1):
        if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
            maxima.append(i)
    if len(maxima) < 2:
        return 1.0

    # Choose top two peaks by density.
    maxima = sorted(maxima, key=lambda i: dens[i], reverse=True)[:2]
    left, right = sorted(maxima)
    if left >= med_idx or right <= med_idx:
        # Peaks not straddling the median -> not a "missing middle" configuration.
        return 1.0

    valley = float(np.min(dens[left : right + 1]))
    peak_min = float(min(dens[left], dens[right]))
    if peak_min <= 0:
        return float("nan")
    return float(valley / peak_min)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float)
    q = q.astype(float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a, b) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _normalized_js_stability(p: np.ndarray, q: np.ndarray) -> float:
    js = _js_divergence(p, q)
    # With natural logs, JS is in [0, ln(2)].
    js_norm = js / math.log(2.0) if math.log(2.0) > 0 else float("nan")
    js_norm = float(np.clip(js_norm, 0.0, 1.0))
    return float(1.0 - js_norm)


def fetch_pbdb_intervals() -> dict[str, dict[str, float]]:
    r = requests.get(PBDB_INTERVALS_URL, params={"scale": "1"}, timeout=60)
    r.raise_for_status()
    records = r.json().get("records") or []
    out: dict[str, dict[str, float]] = {}
    for rec in records:
        name = rec.get("nam")
        eag = rec.get("eag")
        lag = rec.get("lag")
        if not name or eag is None or lag is None:
            continue
        out[str(name)] = {"eag": float(eag), "lag": float(lag)}
    return out


def _interval_bounds(intervals: dict[str, dict[str, float]], label: str) -> tuple[float, float] | None:
    name = str(label).replace("_", " ").strip()
    if name in intervals:
        rec = intervals[name]
        return float(rec["eag"]), float(rec["lag"])
    for prefix in ("Early ", "Late ", "Middle "):
        if name.startswith(prefix):
            base = name[len(prefix) :]
            if base in intervals:
                rec = intervals[base]
                return float(rec["eag"]), float(rec["lag"])
    return None


def load_benson2014_dataset_s1(*, cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cache_path.exists():
        content = cache_path.read_bytes()
    else:
        r = requests.get(PLOS_DATASET_S1_URL, params=PLOS_DATASET_S1_PARAMS, timeout=60)
        r.raise_for_status()
        content = r.content
        _ensure_dir(cache_path.parent)
        cache_path.write_bytes(content)

    xls = pd.ExcelFile(io.BytesIO(content))
    data = pd.read_excel(xls, sheet_name="Data.txt")
    mass = pd.read_excel(xls, sheet_name="Mass estimates")

    # The two sheets are row-aligned but the first column in "Mass estimates" contains non-integer
    # identifiers (e.g., 76.5) that do not safely cast to integer, so we align rows by Taxon and
    # within-taxon order instead of trusting that first column as a stable key.
    data = data.rename(columns={"Unnamed: 0": "data_row_id"})
    mass = mass.rename(columns={"Unnamed: 0": "mass_row_id"})

    data["_taxon_ix"] = data.groupby("Taxon").cumcount()
    mass["_taxon_ix"] = mass.groupby("Taxon").cumcount()

    aligned_mass = data[["Taxon", "_taxon_ix", "data_row_id"]].merge(
        mass[
            [
                "Taxon",
                "_taxon_ix",
                "mass_row_id",
                "Mass 1 /kg (facultative quadrupeds as bipeds)",
                "Mass 2 /kg (facultative quadrupeds as quadrupeds)",
                "Notes",
            ]
        ],
        on=["Taxon", "_taxon_ix"],
        how="left",
        validate="one_to_one",
    )

    data = data.rename(columns={"data_row_id": "specimen_id"}).drop(columns=["_taxon_ix"])
    aligned_mass = aligned_mass.rename(columns={"data_row_id": "specimen_id"}).drop(columns=["_taxon_ix"])

    data["specimen_id"] = pd.to_numeric(data["specimen_id"], errors="coerce").astype("Int64")
    aligned_mass["specimen_id"] = pd.to_numeric(aligned_mass["specimen_id"], errors="coerce").astype("Int64")
    mass = aligned_mass
    return data, mass


def build_body_mass_timebins(
    data: pd.DataFrame,
    mass: pd.DataFrame,
    *,
    intervals: dict[str, dict[str, float]],
    time_bin_myr: float,
    exclude_avialae: bool,
    use_mass_variant: str,
    min_n_per_bin: int,
) -> pd.DataFrame:
    merged = build_body_mass_specimens(
        data,
        mass,
        intervals=intervals,
        time_bin_myr=time_bin_myr,
        exclude_avialae=exclude_avialae,
        use_mass_variant=use_mass_variant,
    )

    rows = []
    for t, g in merged.groupby("time_bin", sort=True):
        x = g["log10_mass_kg"].to_numpy(dtype=float)
        if len(x) < int(min_n_per_bin):
            continue
        bc = _moment_bimodality(x)
        gap = _gap_ratio_hist(x)
        rows.append(
            {
                "time_bin": float(t),
                "n_specimens": int(len(x)),
                "median_log10_mass": float(np.median(x)),
                "mean_log10_mass": float(np.mean(x)),
                "sd_log10_mass": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                "bimodality_coeff": float(bc),
                "gap_ratio_hist": float(gap),
                "exclude_avialae": bool(exclude_avialae),
                "mass_variant": use_mass_variant,
            }
        )

    out = pd.DataFrame(rows).sort_values("time_bin", ascending=False).reset_index(drop=True)
    return out


def build_body_mass_specimens(
    data: pd.DataFrame,
    mass: pd.DataFrame,
    *,
    intervals: dict[str, dict[str, float]],
    time_bin_myr: float,
    exclude_avialae: bool,
    use_mass_variant: str,
) -> pd.DataFrame:
    merged = data.merge(
        mass[
            [
                "specimen_id",
                "Mass 1 /kg (facultative quadrupeds as bipeds)",
                "Mass 2 /kg (facultative quadrupeds as quadrupeds)",
            ]
        ],
        on="specimen_id",
        how="left",
    )

    merged["Juvenile"] = pd.to_numeric(merged["Juvenile"], errors="coerce").fillna(0).astype(int)
    merged = merged[merged["Juvenile"] == 0].copy()

    merged = merged[merged["Clade"].isin(["Theropoda", "Ornithischia", "Sauropodomorpha"])].copy()
    if exclude_avialae:
        merged = merged[merged["Subclade"] != "Avialae"].copy()

    if use_mass_variant == "mass1":
        mcol = "Mass 1 /kg (facultative quadrupeds as bipeds)"
    elif use_mass_variant == "mass2":
        mcol = "Mass 2 /kg (facultative quadrupeds as quadrupeds)"
    else:
        raise ValueError("use_mass_variant must be 'mass1' or 'mass2'")

    merged[mcol] = pd.to_numeric(merged[mcol], errors="coerce")
    merged = merged.dropna(subset=[mcol, "Max_age", "Min_age"]).copy()

    # Convert interval labels to numeric midpoints using bounds.
    max_bounds = merged["Max_age"].apply(lambda s: _interval_bounds(intervals, str(s)))
    min_bounds = merged["Min_age"].apply(lambda s: _interval_bounds(intervals, str(s)))
    merged["max_eag"] = max_bounds.apply(lambda t: t[0] if t else np.nan)
    merged["min_lag"] = min_bounds.apply(lambda t: t[1] if t else np.nan)
    merged["mid_ma"] = (merged["max_eag"] + merged["min_lag"]) / 2.0

    merged = merged.dropna(subset=["mid_ma"]).copy()
    merged["time_bin"] = (merged["mid_ma"] / float(time_bin_myr)).round() * float(time_bin_myr)
    merged["log10_mass_kg"] = np.log10(merged[mcol].astype(float).clip(lower=1e-9))
    merged["exclude_avialae"] = bool(exclude_avialae)
    merged["mass_variant"] = str(use_mass_variant)
    merged = merged.sort_values(["time_bin", "Taxon"], ascending=[False, True]).reset_index(drop=True)
    return merged[
        [
            "specimen_id",
            "Taxon",
            "Clade",
            "Subclade",
            "Subclade_2",
            "Max_age",
            "Min_age",
            "mid_ma",
            "time_bin",
            "log10_mass_kg",
            "exclude_avialae",
            "mass_variant",
        ]
    ].copy()


def load_pbdb_dinosaur_occurrences(
    data_path: str,
    *,
    time_bin_myr: float,
    grid_deg: float,
) -> pd.DataFrame:
    cols = [
        "source_db",
        "occurrence_id",
        "genus",
        "class",
        "mid_ma",
        "lat",
        "lng",
        "paleolat",
        "paleolng",
    ]
    df = pd.read_parquet(data_path, columns=cols)

    df = df[df["source_db"] == "PBDB"].copy()
    df = df.drop_duplicates(subset=["source_db", "occurrence_id"]).copy()

    for c in ["genus", "class"]:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].str.lower().isin({"", "nan", "none", "null"}), c] = pd.NA
    df = df.dropna(subset=["genus", "class", "mid_ma"]).copy()

    for c in ["mid_ma", "lat", "lng", "paleolat", "paleolng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mid_ma"]).copy()

    # Dinosaur proxy in PBDB taxonomy fields.
    df = df[df["class"].isin(["Saurischia", "Ornithischia"])].copy()

    # Paleo coords with fallback.
    df["analysis_lat"] = df["paleolat"].where(df["paleolat"].notna(), df["lat"])
    df["analysis_lng"] = df["paleolng"].where(df["paleolng"].notna(), df["lng"])
    df = df.dropna(subset=["analysis_lat", "analysis_lng"]).copy()

    df["analysis_lat"] = df["analysis_lat"].clip(-90, 90)
    df["analysis_lng"] = df["analysis_lng"].where(df["analysis_lng"] <= 180, df["analysis_lng"] - 360)
    df["analysis_lng"] = df["analysis_lng"].where(df["analysis_lng"] >= -180, df["analysis_lng"] + 360)

    df["time_bin"] = (df["mid_ma"] / float(time_bin_myr)).round() * float(time_bin_myr)
    df["lat_bin"] = (df["analysis_lat"] / float(grid_deg)).round() * float(grid_deg)
    df["lng_bin"] = (df["analysis_lng"] / float(grid_deg)).round() * float(grid_deg)
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))
    return df


def compute_pbdb_stability_timebins(df: pd.DataFrame, *, time_bin_myr: float) -> pd.DataFrame:
    # Richness per grid cell per bin (unique genera).
    cell_richness = (
        df.groupby(["time_bin", "locality"])["genus"]
        .nunique()
        .rename("richness")
        .reset_index()
        .sort_values(["time_bin"], ascending=False)
    )

    # Global centroid per bin (based on localities, to reduce oversampling).
    loc = df.drop_duplicates(subset=["time_bin", "locality"])[["time_bin", "lat_bin", "lng_bin"]]
    glob = (
        loc.groupby("time_bin")
        .agg(
            centroid_lat=("lat_bin", "median"),
            centroid_lng=("lng_bin", lambda s: _circular_mean_deg(s.to_numpy(dtype=float))),
            n_cells=("lat_bin", "size"),
        )
        .reset_index()
        .sort_values("time_bin", ascending=False)
        .reset_index(drop=True)
    )

    # JS stability between consecutive bins.
    bins = sorted(cell_richness["time_bin"].unique(), reverse=True)
    maps: dict[float, dict[tuple[float, float], float]] = {}
    for t in bins:
        sub = cell_richness[cell_richness["time_bin"] == t]
        maps[float(t)] = {tuple(row["locality"]): float(row["richness"]) for _, row in sub.iterrows()}

    rows = []
    for i, t in enumerate(bins):
        t = float(t)
        next_t = float(bins[i + 1]) if i + 1 < len(bins) else float("nan")

        stability_to_next = float("nan")
        if i + 1 < len(bins):
            a = maps[t]
            b = maps[next_t]
            keys = sorted(set(a) | set(b))
            p = np.array([a.get(k, 0.0) for k in keys], dtype=float)
            q = np.array([b.get(k, 0.0) for k in keys], dtype=float)
            stability_to_next = _normalized_js_stability(p, q)

        # Centroid velocity to next (km/Myr).
        if i + 1 < len(bins):
            row_t = glob.loc[glob["time_bin"] == t].iloc[0]
            row_n = glob.loc[glob["time_bin"] == next_t].iloc[0]
            delta = float(t - next_t)
            dist = _haversine_km(
                float(row_t["centroid_lat"]),
                float(row_t["centroid_lng"]),
                float(row_n["centroid_lat"]),
                float(row_n["centroid_lng"]),
            )
            centroid_vel = dist / delta if delta > 0 else float("nan")
        else:
            centroid_vel = float("nan")

        rows.append(
            {
                "time_bin": t,
                "next_time_bin": next_t,
                "stability_to_next": float(stability_to_next),
                "centroid_velocity_km_per_myr": float(centroid_vel),
            }
        )

    out = pd.DataFrame(rows).merge(glob, on="time_bin", how="left")
    out["time_bin_myr"] = float(time_bin_myr)
    return out.sort_values("time_bin", ascending=False).reset_index(drop=True)


def _perm_test_corr(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
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


def _plot_time_series(df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    ax2 = ax1.twinx()

    ax1.plot(df["time_bin"], df["stability_to_next"], color="#1f77b4", marker="o", linewidth=1.5)
    ax1.set_ylabel("Biogeographic stability (1 - JS)")
    ax1.set_xlabel("Time bin (Ma; older → younger)")
    ax1.invert_xaxis()

    ax2.plot(df["time_bin"], df["bimodality_coeff"], color="#d62728", marker="s", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Bimodality coefficient (log10 mass)")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y, "n_specimens"]].dropna().copy()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(d[x], d[y], s=np.clip(d["n_specimens"] * 1.5, 20, 200), alpha=0.75, color="#2ca02c", edgecolors="none")
    # Fit line (unweighted).
    A = np.vstack([d[x].to_numpy(), np.ones(len(d))]).T
    coef, _, _, _ = np.linalg.lstsq(A, d[y].to_numpy(), rcond=None)
    xx = np.linspace(float(d[x].min()), float(d[x].max()), 50)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="black", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="thesis/body_size_stability/output")
    p.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--grid-deg", type=float, default=10.0)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-n-per-bin", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # Fetch interval mapping for stage/epoch names → numeric Ma.
    intervals = fetch_pbdb_intervals()
    (out_dir / "pbdb_intervals_scale1.json").write_text(json.dumps(intervals, indent=2) + "\n")

    # Download/cache Dataset S1.
    cache_path = out_dir / "external" / "benson2014_plosbio_dataset_s1.xls"
    data, mass = load_benson2014_dataset_s1(cache_path=cache_path)

    # Compute body-size time bins (two variants × with/without Avialae).
    body_bins = []
    specimen_bins = []
    for exclude_avialae in (False, True):
        for variant in ("mass1", "mass2"):
            specimens = build_body_mass_specimens(
                data,
                mass,
                intervals=intervals,
                time_bin_myr=float(args.time_bin_myr),
                exclude_avialae=exclude_avialae,
                use_mass_variant=variant,
            )
            specimens_path = out_dir / f"body_mass_specimens_exclAvialae_{int(exclude_avialae)}_{variant}.csv"
            specimens.to_csv(specimens_path, index=False)
            specimen_bins.append(specimens)

            bb = build_body_mass_timebins(
                data,
                mass,
                intervals=intervals,
                time_bin_myr=float(args.time_bin_myr),
                exclude_avialae=exclude_avialae,
                use_mass_variant=variant,
                min_n_per_bin=int(args.min_n_per_bin),
            )
            body_bins.append(bb)
    body = pd.concat(body_bins, ignore_index=True)
    body.to_csv(out_dir / "body_mass_timebins.csv", index=False)
    pd.concat(specimen_bins, ignore_index=True).to_csv(out_dir / "body_mass_specimens_all_variants.csv", index=False)

    # PBDB dinosaur stability.
    occ = load_pbdb_dinosaur_occurrences(
        args.pbdb,
        time_bin_myr=float(args.time_bin_myr),
        grid_deg=float(args.grid_deg),
    )
    pb = compute_pbdb_stability_timebins(occ, time_bin_myr=float(args.time_bin_myr))
    pb.to_csv(out_dir / "pbdb_stability_timebins.csv", index=False)

    # Merge (per variant) and compute correlations + permutation p-values.
    results: list[dict[str, Any]] = []
    merged_paths = []
    for (exclude_avialae, variant), sub in body.groupby(["exclude_avialae", "mass_variant"], sort=False):
        merged = sub.merge(pb, on="time_bin", how="left")
        merged = merged.dropna(subset=["stability_to_next"]).copy()
        merged_path = out_dir / f"merged_timebins_exclAvialae_{int(exclude_avialae)}_{variant}.csv"
        merged.to_csv(merged_path, index=False)
        merged_paths.append(str(merged_path))

        corr_bc = _perm_test_corr(
            merged["stability_to_next"].to_numpy(),
            merged["bimodality_coeff"].to_numpy(),
            permutations=int(args.permutations),
            seed=int(args.seed),
        )
        corr_gap = _perm_test_corr(
            merged["stability_to_next"].to_numpy(),
            merged["gap_ratio_hist"].to_numpy(),
            permutations=int(args.permutations),
            seed=int(args.seed) + 1,
        )

        results.append(
            {
                "exclude_avialae": bool(exclude_avialae),
                "mass_variant": str(variant),
                "n_bins": int(len(merged)),
                "corr_stability_vs_bimodality": corr_bc,
                "corr_stability_vs_gap_ratio": corr_gap,
            }
        )

        # Figures (time series and scatter).
        merged_sorted = merged.sort_values("time_bin", ascending=False)
        _plot_time_series(
            merged_sorted,
            out_path=fig_dir / f"timeseries_exclAvialae_{int(exclude_avialae)}_{variant}.png",
            title=f"Stability vs size bimodality (excl Avialae={exclude_avialae}, {variant})",
        )
        _plot_scatter(
            merged_sorted,
            x="stability_to_next",
            y="bimodality_coeff",
            out_path=fig_dir / f"scatter_stability_bimodality_exclAvialae_{int(exclude_avialae)}_{variant}.png",
            title=f"Stability vs bimodality (excl Avialae={exclude_avialae}, {variant})",
        )
        _plot_scatter(
            merged_sorted,
            x="stability_to_next",
            y="gap_ratio_hist",
            out_path=fig_dir / f"scatter_stability_gapratio_exclAvialae_{int(exclude_avialae)}_{variant}.png",
            title=f"Stability vs gap ratio (excl Avialae={exclude_avialae}, {variant})",
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Write a short summary markdown.
    lines = [
        "# Exploratory results: body-size structure vs biogeographic stability",
        "",
        "This is a first-pass test linking a dinosaur body-mass time series (Benson et al. 2014 Dataset S1) to a PBDB-derived",
        "spatial stability proxy (1 - normalized Jensen–Shannon divergence of dinosaur genus-richness grids between adjacent bins).",
        "",
        f"- Time bin: {float(args.time_bin_myr)} Myr",
        f"- Grid: {float(args.grid_deg)}°",
        f"- Permutation test: {int(args.permutations)} shuffles",
        "",
        "## Correlation summaries",
        "",
        "| Exclude Avialae | Mass variant | n bins | corr(stability,bimodality) | perm-p | corr(stability,gap_ratio) | perm-p |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        bc = r["corr_stability_vs_bimodality"]
        gr = r["corr_stability_vs_gap_ratio"]
        lines.append(
            "| {ex} | {mv} | {n} | {c1:.3f} | {p1:.3g} | {c2:.3f} | {p2:.3g} |".format(
                ex=int(r["exclude_avialae"]),
                mv=r["mass_variant"],
                n=r["n_bins"],
                c1=float(bc["corr"]) if np.isfinite(bc["corr"]) else float("nan"),
                p1=float(bc["p_perm"]) if np.isfinite(bc["p_perm"]) else float("nan"),
                c2=float(gr["corr"]) if np.isfinite(gr["corr"]) else float("nan"),
                p2=float(gr["p_perm"]) if np.isfinite(gr["p_perm"]) else float("nan"),
            )
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Body-mass bins: `{out_dir / 'body_mass_timebins.csv'}`",
            f"- Body-mass specimens (all variants): `{out_dir / 'body_mass_specimens_all_variants.csv'}`",
            f"- Body-mass specimens (per variant): `{out_dir / 'body_mass_specimens_exclAvialae_0_mass1.csv'}` etc.",
            f"- PBDB stability bins: `{out_dir / 'pbdb_stability_timebins.csv'}`",
            f"- Merged per-variant bins: " + ", ".join(f"`{p}`" for p in merged_paths),
            f"- Figures: `{fig_dir}`",
            "",
            "## Interpretation guardrails",
            "",
            "- These correlations are **not causal** and may reflect sampling artifacts in either dataset.",
            "- The stability proxy is PBDB-occurrence-based and can move with outcrop/collection focus.",
            "- Treat any signal as a hypothesis generator; next steps should use independent plate/climate stability series and sampling-aware modeling.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
