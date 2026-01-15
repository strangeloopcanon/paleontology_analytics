from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_name(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _analysis_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
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


def _mutual_information(counts: np.ndarray) -> tuple[float, float, float]:
    """
    Mutual information I(X;Y) for a nonnegative contingency table.
    Returns (mi, hx, hy) in nats.
    """
    counts = counts.astype(float)
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    pxy = counts / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)

    # entropies
    px1 = px[:, 0]
    py1 = py[0, :]
    hx = float(-np.sum(px1[px1 > 0] * np.log(px1[px1 > 0])))
    hy = float(-np.sum(py1[py1 > 0] * np.log(py1[py1 > 0])))

    denom = px @ py
    mask = (pxy > 0) & (denom > 0)
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))
    return mi, hx, hy


def _nmi(mi: float, hx: float, hy: float) -> dict[str, float]:
    if not np.isfinite(mi) or not np.isfinite(hx) or not np.isfinite(hy):
        return {"nmi_sqrt": float("nan"), "nmi_min": float("nan")}
    if hx <= 0 or hy <= 0:
        return {"nmi_sqrt": float("nan"), "nmi_min": float("nan")}
    nmi_sqrt = float(mi / np.sqrt(hx * hy))
    nmi_min = float(mi / float(min(hx, hy)))
    return {"nmi_sqrt": nmi_sqrt, "nmi_min": nmi_min}


def _pca_scores(X: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(float)
    mask = np.all(np.isfinite(X), axis=1)
    if int(np.sum(mask)) < max(6, k + 3):
        return np.full((len(X), k), np.nan), np.full(k, np.nan), np.full((k, X.shape[1]), np.nan)
    Xc = X[mask]
    mu = np.mean(Xc, axis=0)
    sd = np.std(Xc, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Xc - mu) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S**2) / np.sum(S**2)
    kk = min(int(k), Vt.shape[0])
    scores = np.full((len(X), k), np.nan, dtype=float)
    scores[mask, :kk] = U[:, :kk] * S[:kk]
    explained = np.full(k, np.nan, dtype=float)
    explained[:kk] = var[:kk]
    loadings = np.full((k, X.shape[1]), np.nan, dtype=float)
    loadings[:kk, :] = Vt[:kk, :]
    return scores, explained, loadings


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    X = X.astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, fill_value=np.nan, dtype=float)
    if int(np.sum(mask)) < (X.shape[1] + 3):
        return out
    yy = y[mask]
    XX = X[mask]
    A = np.column_stack([np.ones(len(XX)), XX])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    out[mask] = yy - A.dot(beta)
    return out


def _iid_perm_p(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 6:
        return {"corr": float("nan"), "p_perm": float("nan"), "n": float(n)}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(int(seed))
    more = 0
    for _ in range(int(permutations)):
        yp = rng.permutation(y)
        c = float(np.corrcoef(x, yp)[0, 1])
        if abs(c) >= abs(obs):
            more += 1
    p = (more + 1) / (int(permutations) + 1)
    return {"corr": float(obs), "p_perm": float(p), "n": float(n)}


def _circular_shift_p(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict[str, float]:
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 6:
        return {"corr": float("nan"), "p_shift": float("nan"), "n": float(n)}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(int(seed))
    shifts = rng.integers(1, n, size=int(permutations))
    more = 0
    for s in shifts:
        ys = np.roll(y, int(s))
        c = float(np.corrcoef(x, ys)[0, 1])
        if abs(c) >= abs(obs):
            more += 1
    p = (more + 1) / (int(permutations) + 1)
    return {"corr": float(obs), "p_shift": float(p), "n": float(n)}


def _plot_scatter(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> None:
    d = df[[x, y]].dropna()
    if len(d) < 6:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    ax.scatter(d[x], d[y], alpha=0.75, s=40, color="#1f77b4", edgecolors="none")
    A = np.vstack([d[x].to_numpy(), np.ones(len(d))]).T
    coef, *_ = np.linalg.lstsq(A, d[y].to_numpy(), rcond=None)
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
    p.add_argument("--pbdb-extended", default="data/processed/pbdb_occurrences_extended.parquet")
    p.add_argument("--mapping", default="thesis/convergence/output_v3_fullpbdb/ecospace_genus_mapping.csv")
    p.add_argument("--convergence-pairs", default="thesis/convergence/output_v3_fullpbdb/pairwise_sample.csv")
    p.add_argument("--convergence-bins", default="thesis/convergence/output_v3_fullpbdb/timebin_metrics.csv")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--macrostrat", default="data/processed/external/macrostrat/macrostrat_sections_timeseries_bin10.csv")
    p.add_argument("--out", default="thesis/synthesis/output_role_interchangeability_mi_v1")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--grid-deg", type=float, default=10.0)
    p.add_argument("--min-genera-per-region", type=int, default=25)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=77)
    args = p.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    conv_bins = pd.read_csv(args.convergence_bins)
    bins_set = set(conv_bins["time_bin"].astype(float).tolist())

    # Collect the set of localities used by the main marine convergence pipeline, by time bin.
    # This ensures our MI metric is computed on the same spatial units that define the headline result.
    loc_by_bin: dict[float, set[str]] = {float(tb): set() for tb in bins_set}
    pairs_path = Path(args.convergence_pairs)
    chunk_size = 250_000
    for chunk in pd.read_csv(pairs_path, usecols=["time_bin", "loc_a", "loc_b"], chunksize=chunk_size):
        chunk["time_bin"] = pd.to_numeric(chunk["time_bin"], errors="coerce")
        chunk = chunk.dropna(subset=["time_bin"]).copy()
        for tb, g in chunk.groupby("time_bin", sort=False):
            tb = float(tb)
            if tb not in loc_by_bin:
                continue
            s = loc_by_bin[tb]
            s.update(g["loc_a"].dropna().astype(str).tolist())
            s.update(g["loc_b"].dropna().astype(str).tolist())

    # Build an allowlist table of (time_bin, locality_str).
    allow_rows = []
    for tb, locs in loc_by_bin.items():
        for loc in locs:
            allow_rows.append({"time_bin": float(tb), "locality_str": str(loc)})
    allow = pd.DataFrame(allow_rows)
    allow = allow.dropna().drop_duplicates()

    mapping = pd.read_csv(args.mapping)
    mapping["genus"] = mapping["genus"].map(_clean_name)
    mapping["role_id"] = mapping["role_id"].map(_clean_name)
    mapping["jev"] = mapping["jev"].map(_clean_name)
    mapping = mapping.dropna(subset=["genus", "role_id"]).copy()
    mapping = mapping[mapping["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()

    pb = pd.read_parquet(
        args.pbdb_extended,
        columns=[
            "mid_ma",
            "genus",
            "family",
            "order",
            "paleolat",
            "paleolng",
            "lat",
            "lng",
        ],
    )
    pb["genus"] = pb["genus"].map(_clean_name)
    pb["family"] = pb["family"].map(_clean_name)
    pb["order"] = pb["order"].map(_clean_name)
    pb = pb.dropna(subset=["genus", "family", "order", "mid_ma"]).copy()
    pb = _analysis_lat_lng(pb)

    pb["time_bin"] = (pd.to_numeric(pb["mid_ma"], errors="coerce") / float(args.time_bin_myr)).round() * float(
        args.time_bin_myr
    )
    pb = pb[pb["time_bin"].isin(bins_set)].copy()
    pb["lat_bin"] = (pb["analysis_lat"] / float(args.grid_deg)).round() * float(args.grid_deg)
    pb["lng_bin"] = (pb["analysis_lng"] / float(args.grid_deg)).round() * float(args.grid_deg)
    pb["locality"] = list(zip(pb["lat_bin"], pb["lng_bin"]))
    pb["locality_str"] = pb["locality"].map(str)

    df = pb.merge(mapping[["genus", "role_id"]], on="genus", how="inner")

    # Match the convergence pipeline: deduplicate genus within locality×bin to reduce oversampling.
    df = df.drop_duplicates(subset=["time_bin", "locality_str", "genus"]).copy()

    # Convergence pipeline also filters localities by minimum genera per region.
    loc_counts = df.groupby(["time_bin", "locality_str"])["genus"].nunique().rename("n_genera").reset_index()
    loc_counts = loc_counts[loc_counts["n_genera"] >= int(args.min_genera_per_region)].copy()
    df = df.merge(loc_counts[["time_bin", "locality_str"]], on=["time_bin", "locality_str"], how="inner")

    # Restrict to the exact set of localities used in the main convergence pipeline per bin.
    df = df.merge(allow, on=["time_bin", "locality_str"], how="inner")

    # Compute MI/NMI for family↔role and order↔role per bin (two weightings: locality-weighted and genus-presence weighted).
    rows = []
    for tb, g in df.groupby("time_bin", sort=False):
        tb = float(tb)
        n_records = int(len(g))
        n_localities = int(g["locality_str"].nunique())
        n_genera = int(g["genus"].nunique())
        n_roles = int(g["role_id"].nunique())
        n_families = int(g["family"].nunique())
        n_orders = int(g["order"].nunique())

        # Locality-weighted contingency.
        fr = g.groupby(["family", "role_id"]).size().rename("n").reset_index()
        fams = fr["family"].astype(str).tolist()
        roles = fr["role_id"].astype(str).tolist()
        fam_index = {f: i for i, f in enumerate(sorted(set(fams)))}
        role_index = {r: i for i, r in enumerate(sorted(set(roles)))}
        mat = np.zeros((len(fam_index), len(role_index)), dtype=float)
        for _, row in fr.iterrows():
            mat[fam_index[str(row["family"])], role_index[str(row["role_id"])]] = float(row["n"])
        mi_fr, h_fam, h_role = _mutual_information(mat)
        nmis_fr = _nmi(mi_fr, h_fam, h_role)

        # Genus-presence weighting (each genus counts once in the bin).
        gg = g.drop_duplicates(subset=["genus"]).copy()
        fr_g = gg.groupby(["family", "role_id"]).size().rename("n").reset_index()
        fams_g = fr_g["family"].astype(str).tolist()
        roles_g = fr_g["role_id"].astype(str).tolist()
        fam_index_g = {f: i for i, f in enumerate(sorted(set(fams_g)))}
        role_index_g = {r: i for i, r in enumerate(sorted(set(roles_g)))}
        mat_g = np.zeros((len(fam_index_g), len(role_index_g)), dtype=float)
        for _, row in fr_g.iterrows():
            mat_g[fam_index_g[str(row["family"])], role_index_g[str(row["role_id"])]] = float(row["n"])
        mi_fr_g, h_fam_g, h_role_g = _mutual_information(mat_g)
        nmis_fr_g = _nmi(mi_fr_g, h_fam_g, h_role_g)

        # Order ↔ role (genus-presence weighting; usually more stable).
        or_g = gg.groupby(["order", "role_id"]).size().rename("n").reset_index()
        ords = or_g["order"].astype(str).tolist()
        roles_o = or_g["role_id"].astype(str).tolist()
        ord_index = {o: i for i, o in enumerate(sorted(set(ords)))}
        role_index_o = {r: i for i, r in enumerate(sorted(set(roles_o)))}
        mat_or = np.zeros((len(ord_index), len(role_index_o)), dtype=float)
        for _, row in or_g.iterrows():
            mat_or[ord_index[str(row["order"])], role_index_o[str(row["role_id"])]] = float(row["n"])
        mi_or_g, h_ord_g, h_role_or_g = _mutual_information(mat_or)
        nmis_or_g = _nmi(mi_or_g, h_ord_g, h_role_or_g)

        rows.append(
            {
                "time_bin": tb,
                "n_records": n_records,
                "n_localities": n_localities,
                "n_genera": n_genera,
                "n_roles": n_roles,
                "n_families": n_families,
                "n_orders": n_orders,
                "mi_family_role_locality": float(mi_fr),
                "h_family_locality": float(h_fam),
                "h_role_locality": float(h_role),
                "nmi_family_role_locality_sqrt": float(nmis_fr["nmi_sqrt"]),
                "nmi_family_role_locality_min": float(nmis_fr["nmi_min"]),
                "mi_family_role_genus": float(mi_fr_g),
                "h_family_genus": float(h_fam_g),
                "h_role_genus": float(h_role_g),
                "nmi_family_role_genus_sqrt": float(nmis_fr_g["nmi_sqrt"]),
                "nmi_family_role_genus_min": float(nmis_fr_g["nmi_min"]),
                "mi_order_role_genus": float(mi_or_g),
                "nmi_order_role_genus_sqrt": float(nmis_or_g["nmi_sqrt"]),
                "nmi_order_role_genus_min": float(nmis_or_g["nmi_min"]),
            }
        )

    mi_df = pd.DataFrame(rows).sort_values("time_bin", ascending=False).reset_index(drop=True)
    # Avoid column collisions when merging with the convergence-derived base table.
    mi_df = mi_df.rename(columns={"n_localities": "mi_n_localities"})
    mi_df.to_csv(out_dir / "timebin_role_interchangeability.csv", index=False)

    # Merge with forcing + sampling proxies (reuse the already merged convergence table for these bins).
    base = pd.read_csv("thesis/synthesis/output_convergence_sampling_autocorr_fullpbdb_macrostrat_pca_v1/merged.csv")
    merged = base.merge(mi_df, on="time_bin", how="left")
    merged = merged.dropna(subset=["delta_from_prev_T_field_meanabs", "nmi_family_role_genus_sqrt"]).copy()

    # Sampling PCA index (handles collinearity between PBDB sampling proxies and Macrostrat proxies).
    sampling_features = np.column_stack(
        [
            np.log1p(merged["n_localities"].to_numpy(dtype=float)),
            np.log1p(merged["marine_n_collections"].to_numpy(dtype=float)),
            np.log1p(merged["marine_n_occurrences"].to_numpy(dtype=float)),
            np.log1p(merged["macro_col_area_sum"].to_numpy(dtype=float)),
            np.log1p(merged["macro_n_sections"].to_numpy(dtype=float)),
        ]
    )
    pcs, pc_expl, pc_load = _pca_scores(sampling_features, k=2)
    merged["sampling_pc1"] = pcs[:, 0]
    merged["sampling_pc2"] = pcs[:, 1]
    (out_dir / "sampling_pca.json").write_text(
        json.dumps(
            {
                "feature_names": [
                    "log1p(n_localities)",
                    "log1p(marine_n_collections)",
                    "log1p(marine_n_occurrences)",
                    "log1p(macro_col_area_sum)",
                    "log1p(macro_n_sections)",
                ],
                "explained_variance": [float(x) for x in pc_expl],
                "loadings": [[float(v) for v in row] for row in pc_load],
            },
            indent=2,
        )
        + "\n"
    )

    merged.to_csv(out_dir / "merged.csv", index=False)

    # Hypothesis tests: volatility vs (1 - NMI) = interchangeability.
    merged["interchangeability_family_role_genus"] = 1.0 - merged["nmi_family_role_genus_sqrt"]
    y = merged["interchangeability_family_role_genus"].to_numpy(dtype=float).astype(float)
    v = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    t = merged["time_bin"].to_numpy(dtype=float)
    prov = merged["provinciality"].to_numpy(dtype=float)
    pc1 = merged["sampling_pc1"].to_numpy(dtype=float)
    pc2 = merged["sampling_pc2"].to_numpy(dtype=float)

    configs = [
        ("control_time", np.column_stack([t])),
        ("control_time_pc1", np.column_stack([t, pc1])),
        ("control_time_pc12", np.column_stack([t, pc1, pc2])),
        ("control_time_pc12_prov", np.column_stack([t, pc1, pc2, prov])),
    ]

    results: dict[str, Any] = {"n_bins": int(len(merged))}
    for i, (name, ctrl) in enumerate(configs):
        rx = _residualize(v, ctrl)
        ry = _residualize(y, ctrl)
        results[f"{name}_iid_perm"] = _iid_perm_p(rx, ry, permutations=int(args.permutations), seed=int(args.seed) + i)
        results[f"{name}_circular_shift"] = _circular_shift_p(
            rx, ry, permutations=int(args.permutations), seed=int(args.seed) + 100 + i
        )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    _plot_scatter(
        merged,
        x="delta_from_prev_T_field_meanabs",
        y="nmi_family_role_genus_sqrt",
        out_path=fig_dir / "scatter_volatility_vs_nmi_family_role.png",
        title="Volatility vs taxon↔role association (NMI; genus-weighted)",
    )
    _plot_scatter(
        merged,
        x="delta_from_prev_T_field_meanabs",
        y="interchangeability_family_role_genus",
        out_path=fig_dir / "scatter_volatility_vs_interchangeability.png",
        title="Volatility vs role interchangeability (1 - NMI)",
    )

    def _fmt(entry: dict[str, Any] | None, p_key: str) -> str:
        if not entry:
            return "corr=nan, p=nan, n=0"
        return "corr={c:.3f}, p={p:.3g}, n={n}".format(
            c=float(entry.get("corr", float("nan"))),
            p=float(entry.get(p_key, float("nan"))),
            n=int(entry.get("n") or 0),
        )

    lines = [
        "# Role interchangeability under volatility (MI/NMI; first pass)",
        "",
        "Hypothesis: climate volatility increases taxon↔role interchangeability (roles become less clade-specific).",
        "",
        "We compute per-bin taxon↔role association strength using mutual information (MI) between `family` (or `order`) and `role_id`.",
        "Interchangeability index = `1 - NMI` (higher = roles are less taxon-specific).",
        "",
        "Spatial scope: same bins + localities as the main marine convergence analysis:",
        f"- bins: `{Path(args.convergence_bins)}`",
        f"- locality allowlist built from: `{pairs_path}`",
        "",
        "Inputs:",
        f"- PBDB occurrences: `{Path(args.pbdb_extended)}`",
        f"- PBDB ecospace mapping: `{Path(args.mapping)}`",
        f"- Earth-system forcing: `{Path(args.earth)}`",
        f"- Macrostrat proxies: `{Path(args.macrostrat)}`",
        "",
        "Taxon↔role metric used for the main test:",
        "- `1 - nmi_family_role_genus_sqrt` (genus-presence weighting; families vs roles).",
        "",
        "Sampling control:",
        f"- sampling PCA PC1 explained variance: {float(pc_expl[0]):.3f}",
        "",
        "## Partial correlation tests (volatility vs interchangeability)",
        "",
        "- IID permutation p-values shuffle residuals (exchangeable bins).",
        "- Circular-shift p-values preserve autocorrelation structure (time-ordered bins).",
        "",
    ]
    for name, _ in configs:
        iid = results.get(f"{name}_iid_perm", {})
        shift = results.get(f"{name}_circular_shift", {})
        lines.append(f"- {name}: iid({_fmt(iid,'p_perm')}); shift({_fmt(shift,'p_shift')})")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- MI time bins: `{out_dir / 'timebin_role_interchangeability.csv'}`",
            f"- Merged table: `{out_dir / 'merged.csv'}`",
            f"- Stats: `{out_dir / 'analysis_results.json'}`",
            f"- Sampling PCA: `{out_dir / 'sampling_pca.json'}`",
            f"- Figures: `{fig_dir}`",
            "",
            "## Notes",
            "",
            "- This is a bin-level analysis; publication-grade inference should move to a pair-level / hierarchical model and a sampling-aware MI estimator if needed.",
            "- `NMI` is sensitive to the taxonomic level chosen; `order`-level metrics are included in `timebin_role_interchangeability.csv` for comparison.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
