from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon


PBDB_TAXA_LIST = "https://paleobiodb.org/data1.2/taxa/list.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_name(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _analysis_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer paleo coords with fallback; normalize longitude to [-180, 180].
    out = df.copy()
    out["analysis_lat"] = out["paleolat"].where(out["paleolat"].notna(), out["lat"])
    out["analysis_lng"] = out["paleolng"].where(out["paleolng"].notna(), out["lng"])
    out = out.dropna(subset=["analysis_lat", "analysis_lng"]).copy()
    out["analysis_lat"] = pd.to_numeric(out["analysis_lat"], errors="coerce").clip(-90, 90)
    out["analysis_lng"] = pd.to_numeric(out["analysis_lng"], errors="coerce")
    out["analysis_lng"] = out["analysis_lng"].where(out["analysis_lng"] <= 180, out["analysis_lng"] - 360)
    out["analysis_lng"] = out["analysis_lng"].where(out["analysis_lng"] >= -180, out["analysis_lng"] + 360)
    out = out.dropna(subset=["analysis_lat", "analysis_lng"]).copy()
    return out


def _js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    # SciPy returns JS distance in [0, 1] when base=2 (default).
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p.astype(float) / float(p.sum())
    q = q.astype(float) / float(q.sum())
    d = float(jensenshannon(p, q))
    if not np.isfinite(d):
        return float("nan")
    return float(1.0 - np.clip(d, 0.0, 1.0))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter / union) if union else float("nan")


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


def _residualize_1d(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    x = x.astype(float)
    mask = np.isfinite(y) & np.isfinite(x)
    yy = y[mask]
    xx = x[mask]
    if len(yy) < 3:
        out = np.full_like(y, fill_value=np.nan, dtype=float)
        out[mask] = np.nan
        return out
    A = np.column_stack([np.ones(len(xx)), xx])
    coef, _, _, _ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A.dot(coef)
    out = np.full_like(y, fill_value=np.nan, dtype=float)
    out[mask] = resid
    return out


def _partial_corr_perm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    control: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    rx = _residualize_1d(x, control)
    ry = _residualize_1d(y, control)
    return _perm_test_corr(rx, ry, permutations=int(permutations), seed=int(seed))


def _fit_ols_1d(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    A = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
    return {"intercept": intercept, "slope": slope, "r2": r2}


@dataclass(frozen=True)
class EcospaceRecord:
    genus: str
    jev: str | None  # environment (e.g., marine/terrestrial)
    jdt: str | None  # diet
    jmo: str | None  # motility
    jlh: str | None  # life habit

    @property
    def role_id(self) -> str | None:
        parts = [self.jdt, self.jmo, self.jlh]
        if any(p is None for p in parts):
            return None
        return "|".join(parts)


def _fetch_pbdb_taxa_ecospace(names: list[str], *, rank: str = "genus") -> list[dict[str, Any]]:
    params = {
        "name": ",".join(names),
        "rank": rank,
        "show": "ecospace",
    }
    r = requests.get(PBDB_TAXA_LIST, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("records") or []


def fetch_ecospace_genus_mapping(
    genera: list[str],
    *,
    cache_path: Path,
    batch_size: int = 100,
    sleep_s: float = 0.15,
) -> dict[str, EcospaceRecord]:
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    pending = [g for g in genera if g not in cache]
    print(f"PBDB ecospace: {len(cache)} cached, {len(pending)} to fetch")

    for i in range(0, len(pending), int(batch_size)):
        batch = pending[i : i + int(batch_size)]
        recs = _fetch_pbdb_taxa_ecospace(batch, rank="genus")
        # PBDB returns a flat list; index by returned name.
        for rec in recs:
            nam = _clean_name(rec.get("nam"))
            if not nam:
                continue
            cache[nam] = rec

        # Record explicit misses so we don't refetch forever.
        for g in batch:
            if g not in cache:
                cache[g] = None

        # Persist frequently for resilience.
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
        if sleep_s > 0:
            time.sleep(float(sleep_s))

    out: dict[str, EcospaceRecord] = {}
    for g in genera:
        rec = cache.get(g)
        if not rec:
            continue
        out[g] = EcospaceRecord(
            genus=g,
            jev=_clean_name(rec.get("jev")),
            jdt=_clean_name(rec.get("jdt")),
            jmo=_clean_name(rec.get("jmo")),
            jlh=_clean_name(rec.get("jlh")),
        )
    return out


def _chunked(it: Iterable[Any], size: int) -> Iterable[list[Any]]:
    buf: list[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def compute_timebin_metrics(
    occ: pd.DataFrame,
    traits: dict[str, EcospaceRecord],
    *,
    time_bin_myr: float,
    grid_deg: float,
    min_genera_per_region: int,
    max_pairs_per_bin: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = occ.copy()
    df["genus"] = df["genus"].map(_clean_name)
    df = df.dropna(subset=["genus", "mid_ma"]).copy()

    df = _analysis_lat_lng(df)
    df["time_bin"] = (pd.to_numeric(df["mid_ma"], errors="coerce") / float(time_bin_myr)).round() * float(time_bin_myr)
    df["lat_bin"] = (df["analysis_lat"] / float(grid_deg)).round() * float(grid_deg)
    df["lng_bin"] = (df["analysis_lng"] / float(grid_deg)).round() * float(grid_deg)
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))

    # Attach ecospace traits.
    tdf = pd.DataFrame(
        [
            {
                "genus": r.genus,
                "jev": r.jev,
                "jdt": r.jdt,
                "jmo": r.jmo,
                "jlh": r.jlh,
                "role_id": r.role_id,
            }
            for r in traits.values()
        ]
    )
    df = df.merge(tdf, on="genus", how="left")

    # Focus on marine (ecospace-coded).
    df = df[df["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()
    df = df.dropna(subset=["role_id"]).copy()

    # De-duplicate within locality×bin by genus to reduce oversampling.
    df = df.drop_duplicates(subset=["time_bin", "locality", "genus"]).copy()

    # Build per locality: genus set and role counts.
    genus_sets = (
        df.groupby(["time_bin", "locality"])["genus"]
        .agg(lambda s: set(s.astype(str)))
        .rename("genus_set")
        .reset_index()
    )
    genus_counts = (
        df.groupby(["time_bin", "locality"])["genus"]
        .nunique()
        .rename("n_genera")
        .reset_index()
    )
    genus_sets = genus_sets.merge(genus_counts, on=["time_bin", "locality"], how="left")

    role_counts = (
        df.groupby(["time_bin", "locality", "role_id"])["genus"]
        .nunique()
        .rename("n_genera_role")
        .reset_index()
    )

    role_sets = (
        df.groupby(["time_bin", "locality"])["role_id"]
        .agg(lambda s: set(s.astype(str)))
        .rename("role_set")
        .reset_index()
    )

    # Convenience: role list for consistent vectors.
    all_roles = sorted(role_counts["role_id"].unique())
    role_index = {r: i for i, r in enumerate(all_roles)}

    # Precompute per bin: locality -> role vector
    vectors: dict[tuple[float, tuple[float, float]], np.ndarray] = {}
    for (t, loc), sub in role_counts.groupby(["time_bin", "locality"], sort=False):
        v = np.zeros(len(all_roles), dtype=float)
        for _, row in sub.iterrows():
            rid = row["role_id"]
            v[role_index[rid]] = float(row["n_genera_role"])
        vectors[(float(t), tuple(loc))] = v

    role_set_map: dict[tuple[float, tuple[float, float]], set[str]] = {}
    for _, row in role_sets.iterrows():
        role_set_map[(float(row["time_bin"]), tuple(row["locality"]))] = row["role_set"]

    bins = sorted(genus_sets["time_bin"].unique(), reverse=True)
    metrics_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    # Global turnover (between consecutive bins) computed on marine genera.
    global_sets: dict[float, set[str]] = {}
    for t in bins:
        global_sets[float(t)] = set(df.loc[df["time_bin"] == t, "genus"].astype(str).unique())

    def global_turnover(prev_bin: float | None, cur_bin: float) -> float:
        if prev_bin is None:
            return float("nan")
        a = global_sets.get(float(prev_bin), set())
        b = global_sets.get(float(cur_bin), set())
        jac = _jaccard_similarity(a, b)
        if not np.isfinite(jac):
            return float("nan")
        return float(1.0 - jac)

    for i, t in enumerate(bins):
        t = float(t)
        prev_t = float(bins[i - 1]) if i - 1 >= 0 else None  # older bin
        next_t = float(bins[i + 1]) if i + 1 < len(bins) else None  # younger bin

        # Candidate localities in this bin.
        loc_df = genus_sets[genus_sets["time_bin"] == t].copy()
        loc_df = loc_df[loc_df["n_genera"] >= int(min_genera_per_region)].copy()
        localities = list(loc_df["locality"])

        n_loc = len(localities)
        if n_loc < 6:
            continue

        # Prepare per-locality objects.
        loc_sets = {tuple(loc): loc_df.loc[loc_df["locality"] == loc, "genus_set"].iloc[0] for loc in localities}
        loc_vecs = {tuple(loc): vectors[(t, tuple(loc))] for loc in localities if (t, tuple(loc)) in vectors}

        # Pairwise similarities.
        pairs = []
        for a_i in range(n_loc - 1):
            loc_a = tuple(localities[a_i])
            for b_i in range(a_i + 1, n_loc):
                loc_b = tuple(localities[b_i])
                pairs.append((loc_a, loc_b))

        if len(pairs) > int(max_pairs_per_bin):
            idx = rng.choice(len(pairs), size=int(max_pairs_per_bin), replace=False)
            pairs = [pairs[j] for j in idx]

        func_sims = []
        func_role_sims = []
        tax_sims = []
        for loc_a, loc_b in pairs:
            va = loc_vecs.get(loc_a)
            vb = loc_vecs.get(loc_b)
            if va is None or vb is None:
                continue
            func = _js_similarity(va, vb)
            roles_a = role_set_map.get((t, tuple(loc_a)), set())
            roles_b = role_set_map.get((t, tuple(loc_b)), set())
            func_role = _jaccard_similarity(roles_a, roles_b)
            tax = _jaccard_similarity(loc_sets.get(loc_a, set()), loc_sets.get(loc_b, set()))
            if not np.isfinite(func) or not np.isfinite(func_role) or not np.isfinite(tax):
                continue
            func_sims.append(func)
            func_role_sims.append(func_role)
            tax_sims.append(tax)
            pair_rows.append(
                {
                    "time_bin": t,
                    "loc_a": str(loc_a),
                    "loc_b": str(loc_b),
                    "functional_similarity_js": float(func),
                    "functional_similarity_roles_jaccard": float(func_role),
                    "taxonomic_similarity": float(tax),
                }
            )

        if len(func_sims) < 200:
            continue

        mean_func = float(np.mean(func_sims))
        mean_func_roles = float(np.mean(func_role_sims)) if func_role_sims else float("nan")
        mean_tax = float(np.mean(tax_sims))
        provinciality = float(1.0 - mean_tax)

        metrics_rows.append(
            {
                "time_bin": t,
                "n_localities": int(n_loc),
                "n_pairs": int(len(func_sims)),
                "mean_functional_similarity_js": mean_func,
                "mean_functional_similarity_roles_jaccard": mean_func_roles,
                "mean_taxonomic_similarity": mean_tax,
                "provinciality": provinciality,
                # turnover entering this bin (prev older -> current)
                "global_turnover_from_prev": global_turnover(prev_t, t),
                # turnover leaving this bin (current -> next younger)
                "global_turnover_to_next": global_turnover(t, next_t) if next_t is not None else float("nan"),
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    metrics = pd.DataFrame(metrics_rows).sort_values("time_bin", ascending=False).reset_index(drop=True)

    if len(pairwise) == 0 or len(metrics) == 0:
        return metrics, pairwise

    # Fit global relationship: functional similarity ~ taxonomic similarity (OLS via NumPy).
    x = pairwise["taxonomic_similarity"].to_numpy(dtype=float)
    fit_js = _fit_ols_1d(x, pairwise["functional_similarity_js"].to_numpy(dtype=float))
    pairwise["functional_js_pred"] = fit_js["intercept"] + fit_js["slope"] * x
    pairwise["functional_js_residual"] = pairwise["functional_similarity_js"] - pairwise["functional_js_pred"]

    fit_roles = _fit_ols_1d(x, pairwise["functional_similarity_roles_jaccard"].to_numpy(dtype=float))
    pairwise["functional_roles_pred"] = fit_roles["intercept"] + fit_roles["slope"] * x
    pairwise["functional_roles_residual"] = (
        pairwise["functional_similarity_roles_jaccard"] - pairwise["functional_roles_pred"]
    )

    # Also fit per-bin coupling models (slope/intercept vary by time bin) for tests of forcing-modulated decoupling.
    bin_rows = []
    for tb, g in pairwise.groupby("time_bin", sort=False):
        xg = g["taxonomic_similarity"].to_numpy(dtype=float)
        js = g["functional_similarity_js"].to_numpy(dtype=float)
        roles = g["functional_similarity_roles_jaccard"].to_numpy(dtype=float)
        f_js = _fit_ols_1d(xg, js)
        f_roles = _fit_ols_1d(xg, roles)
        bin_rows.append(
            {
                "time_bin": float(tb),
                "bin_js_intercept": float(f_js["intercept"]),
                "bin_js_slope_tax_to_func": float(f_js["slope"]),
                "bin_js_r2": float(f_js["r2"]),
                "bin_roles_intercept": float(f_roles["intercept"]),
                "bin_roles_slope_tax_to_func": float(f_roles["slope"]),
                "bin_roles_r2": float(f_roles["r2"]),
            }
        )
    bin_fit = pd.DataFrame(bin_rows)

    # Summarize residual per bin as "functional excess similarity".
    res = pairwise.groupby("time_bin")[["functional_js_residual", "functional_roles_residual"]].mean().reset_index()
    res = res.rename(
        columns={
            "functional_js_residual": "functional_excess_similarity_js",
            "functional_roles_residual": "functional_excess_similarity_roles_jaccard",
        }
    )
    metrics = metrics.merge(res, on="time_bin", how="left")
    metrics = metrics.merge(bin_fit, on="time_bin", how="left")
    metrics["model_js_intercept"] = float(fit_js["intercept"])
    metrics["model_js_slope_tax_to_func"] = float(fit_js["slope"])
    metrics["model_js_r2"] = float(fit_js["r2"])
    metrics["model_roles_intercept"] = float(fit_roles["intercept"])
    metrics["model_roles_slope_tax_to_func"] = float(fit_roles["slope"])
    metrics["model_roles_r2"] = float(fit_roles["r2"])
    return metrics, pairwise


def _plot_time_series(df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.plot(df["time_bin"], df["functional_excess_similarity_js"], marker="o", linewidth=1.6, color="#2ca02c")
    ax.set_xlabel("Time bin (Ma; older → younger)")
    ax.set_ylabel("Functional excess similarity (JS residual)")
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.plot(df["time_bin"], df["global_turnover_from_prev"], marker="s", linewidth=1.3, color="#d62728", alpha=0.7)
    ax2.set_ylabel("Global turnover (from prev bin)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y, "n_localities"]].dropna().copy()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.3, 4.9))
    ax.scatter(d[x], d[y], s=np.clip(d["n_localities"] * 2.0, 20, 200), alpha=0.75, color="#1f77b4", edgecolors="none")
    A = np.vstack([d[x].to_numpy(), np.ones(len(d))]).T
    coef, _, _, _ = np.linalg.lstsq(A, d[y].to_numpy(), rcond=None)
    xx = np.linspace(float(d[x].min()), float(d[x].max()), 60)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, color="black", linewidth=1.2, alpha=0.75)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="thesis/convergence/output")
    p.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--grid-deg", type=float, default=10.0)
    p.add_argument("--min-occ-per-genus", type=int, default=5)
    p.add_argument("--min-genera-per-region", type=int, default=25)
    p.add_argument("--max-pairs-per-bin", type=int, default=30000)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(cache_dir)

    # Load PBDB occurrences from repo.
    cols = [
        "source_db",
        "occurrence_id",
        "mid_ma",
        "lat",
        "lng",
        "paleolat",
        "paleolng",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "environment",
    ]
    occ = pd.read_parquet(args.pbdb, columns=cols)
    occ = occ[occ["source_db"] == "PBDB"].drop_duplicates(subset=["source_db", "occurrence_id"]).copy()
    occ["genus"] = occ["genus"].map(_clean_name)
    occ = occ.dropna(subset=["genus"]).copy()

    # Restrict genus list to reduce API load.
    genus_counts = occ["genus"].value_counts()
    genera = genus_counts[genus_counts >= int(args.min_occ_per_genus)].index.astype(str).tolist()

    cache_path = cache_dir / "pbdb_ecospace_genus_cache.json"
    traits = fetch_ecospace_genus_mapping(genera, cache_path=cache_path)

    mapping_rows = []
    for g, r in traits.items():
        mapping_rows.append(
            {
                "genus": g,
                "jev": r.jev,
                "jdt": r.jdt,
                "jmo": r.jmo,
                "jlh": r.jlh,
                "role_id": r.role_id,
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(out_dir / "ecospace_genus_mapping.csv", index=False)

    metrics, pairwise = compute_timebin_metrics(
        occ,
        traits,
        time_bin_myr=float(args.time_bin_myr),
        grid_deg=float(args.grid_deg),
        min_genera_per_region=int(args.min_genera_per_region),
        max_pairs_per_bin=int(args.max_pairs_per_bin),
        seed=int(args.seed),
    )
    metrics.to_csv(out_dir / "timebin_metrics.csv", index=False)
    pairwise.to_csv(out_dir / "pairwise_sample.csv", index=False)

    # Hypothesis tests (bin-level).
    results: dict[str, Any] = {}
    if len(metrics) >= 6 and "functional_excess_similarity_js" in metrics.columns:
        x1 = metrics["global_turnover_from_prev"].to_numpy(dtype=float)
        y_js = metrics["functional_excess_similarity_js"].to_numpy(dtype=float)
        y_roles = metrics["functional_excess_similarity_roles_jaccard"].to_numpy(dtype=float)
        results["H1_turnover_vs_convergence_js"] = _perm_test_corr(
            x1,
            y_js,
            permutations=int(args.permutations),
            seed=int(args.seed),
        )
        results["H1_turnover_vs_convergence_roles"] = _perm_test_corr(
            x1,
            y_roles,
            permutations=int(args.permutations),
            seed=int(args.seed) + 1,
        )

        x2 = metrics["provinciality"].to_numpy(dtype=float)
        results["H2_provinciality_vs_convergence_js"] = _perm_test_corr(
            x2,
            y_js,
            permutations=int(args.permutations),
            seed=int(args.seed) + 2,
        )
        results["H2_provinciality_vs_convergence_roles"] = _perm_test_corr(
            x2,
            y_roles,
            permutations=int(args.permutations),
            seed=int(args.seed) + 3,
        )

        # Volatility proxy (same as turnover-from-prev for now; later can replace with independent series).
        results["H3_volatility_vs_convergence_js"] = results["H1_turnover_vs_convergence_js"]
        results["H3_volatility_vs_convergence_roles"] = results["H1_turnover_vs_convergence_roles"]

        # Strong long-term trend check (convergence vs time).
        t = metrics["time_bin"].to_numpy(dtype=float)
        results["trend_time_vs_convergence_js"] = _perm_test_corr(t, y_js, permutations=int(args.permutations), seed=int(args.seed) + 10)
        results["trend_time_vs_convergence_roles"] = _perm_test_corr(
            t, y_roles, permutations=int(args.permutations), seed=int(args.seed) + 11
        )

        # Partial correlations controlling for time (to avoid trend-driven answers).
        results["H1_partial_turnover_vs_convergence_js_control_time"] = _partial_corr_perm(
            x1, y_js, control=t, permutations=int(args.permutations), seed=int(args.seed) + 20
        )
        results["H2_partial_provinciality_vs_convergence_js_control_time"] = _partial_corr_perm(
            x2, y_js, control=t, permutations=int(args.permutations), seed=int(args.seed) + 21
        )
        results["H1_partial_turnover_vs_convergence_roles_control_time"] = _partial_corr_perm(
            x1, y_roles, control=t, permutations=int(args.permutations), seed=int(args.seed) + 22
        )
        results["H2_partial_provinciality_vs_convergence_roles_control_time"] = _partial_corr_perm(
            x2, y_roles, control=t, permutations=int(args.permutations), seed=int(args.seed) + 23
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Figures.
    if len(metrics) > 0:
        _plot_time_series(
            metrics.sort_values("time_bin", ascending=False),
            out_path=fig_dir / "timeseries_convergence_turnover.png",
            title="Functional excess similarity vs turnover (marine ecospace; PBDB)",
        )
        _plot_scatter(
            metrics,
            x="global_turnover_from_prev",
            y="functional_excess_similarity_js",
            out_path=fig_dir / "scatter_turnover_convergence.png",
            title="H1/H3: turnover vs convergence (JS residual)",
        )
        _plot_scatter(
            metrics,
            x="provinciality",
            y="functional_excess_similarity_js",
            out_path=fig_dir / "scatter_provinciality_convergence.png",
            title="H2: provinciality vs convergence (JS residual)",
        )
        _plot_scatter(
            metrics,
            x="global_turnover_from_prev",
            y="functional_excess_similarity_roles_jaccard",
            out_path=fig_dir / "scatter_turnover_convergence_roles.png",
            title="H1/H3: turnover vs convergence (role Jaccard residual)",
        )
        _plot_scatter(
            metrics,
            x="provinciality",
            y="functional_excess_similarity_roles_jaccard",
            out_path=fig_dir / "scatter_provinciality_convergence_roles.png",
            title="H2: provinciality vs convergence (role Jaccard residual)",
        )

    # Summary markdown.
    lines = [
        "# Exploratory results: functional convergence using PBDB ecospace",
        "",
        "This run uses PBDB occurrences (from the repo parquet) combined with PBDB taxon ecospace annotations (diet/motility/life habit/environment).",
        "",
        f"- Time bin: {float(args.time_bin_myr)} Myr",
        f"- Grid: {float(args.grid_deg)}°",
        f"- Genus inclusion: ≥ {int(args.min_occ_per_genus)} PBDB occurrences in the repo dataset",
        f"- Locality inclusion: ≥ {int(args.min_genera_per_region)} unique genera per locality×bin (after de-duplication)",
        "",
        "## Hypotheses (bin-level tests, permutation p-values)",
        "",
    ]
    if results:
        h1_js = results.get("H1_turnover_vs_convergence_js", {})
        h1_roles = results.get("H1_turnover_vs_convergence_roles", {})
        h2_js = results.get("H2_provinciality_vs_convergence_js", {})
        h2_roles = results.get("H2_provinciality_vs_convergence_roles", {})
        t_js = results.get("trend_time_vs_convergence_js", {})
        t_roles = results.get("trend_time_vs_convergence_roles", {})
        h2_partial = results.get("H2_partial_provinciality_vs_convergence_js_control_time", {})
        h1_partial = results.get("H1_partial_turnover_vs_convergence_js_control_time", {})
        lines.extend(
            [
                f"- H1 (post-perturbation proxy; JS residual): corr(turnover_from_prev, functional_excess_similarity_js) = {h1_js.get('corr'):.3f}, perm-p = {h1_js.get('p_perm'):.3g}, n = {int(h1_js.get('n') or 0)}",
                f"- H1 (post-perturbation proxy; role Jaccard residual): corr(turnover_from_prev, functional_excess_similarity_roles_jaccard) = {h1_roles.get('corr'):.3f}, perm-p = {h1_roles.get('p_perm'):.3g}, n = {int(h1_roles.get('n') or 0)}",
                f"- H2 (fragmentation/provinciality; JS residual): corr(provinciality, functional_excess_similarity_js) = {h2_js.get('corr'):.3f}, perm-p = {h2_js.get('p_perm'):.3g}, n = {int(h2_js.get('n') or 0)}",
                f"- H2 (fragmentation/provinciality; role Jaccard residual): corr(provinciality, functional_excess_similarity_roles_jaccard) = {h2_roles.get('corr'):.3f}, perm-p = {h2_roles.get('p_perm'):.3g}, n = {int(h2_roles.get('n') or 0)}",
                f"- Trend: corr(time_bin, functional_excess_similarity_js) = {t_js.get('corr'):.3f}, perm-p = {t_js.get('p_perm'):.3g}",
                f"- Trend: corr(time_bin, functional_excess_similarity_roles_jaccard) = {t_roles.get('corr'):.3f}, perm-p = {t_roles.get('p_perm'):.3g}",
                f"- Partial (controls time): corr(provinciality, convergence_js | time) = {h2_partial.get('corr'):.3f}, perm-p = {h2_partial.get('p_perm'):.3g}",
                f"- Partial (controls time): corr(turnover_from_prev, convergence_js | time) = {h1_partial.get('corr'):.3f}, perm-p = {h1_partial.get('p_perm'):.3g}",
                "- H3 (volatility): currently uses the same turnover proxy as H1; will be re-tested with an independent climate/paleogeography volatility series.",
            ]
        )
    else:
        lines.append("- Not enough bins to compute stable metrics under the current thresholds.")

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Ecospace mapping: `{out_dir / 'ecospace_genus_mapping.csv'}`",
            f"- Bin metrics: `{out_dir / 'timebin_metrics.csv'}`",
            f"- Pair sample: `{out_dir / 'pairwise_sample.csv'}`",
            f"- Figures: `{fig_dir}`",
            "",
            "## Interpretation guardrails",
            "",
            "- Ecospace annotations have missingness and may not be uniformly curated across clades/time.",
            "- PBDB occurrences reflect sampling/rock availability; treat this as a hypothesis generator unless sampling and independent forcing are incorporated.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
