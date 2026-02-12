"""Terrestrial convergence pilot using PBDB tetrapod data.

Tests whether the volatility-convergence signal extends to terrestrial
vertebrates (tetrapods). Even a negative result is informative.
Uses PBDB ecospace annotations for tetrapod genera with terrestrial jev.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clean(x: object) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s and s.lower() not in {"nan", "none", "null"} else None


def _js_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    p, q = a / a.sum(), b / b.sum()
    d = float(jensenshannon(p, q))
    return float(1.0 - np.clip(d, 0, 1)) if np.isfinite(d) else float("nan")


def _jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return float(len(a & b) / u) if u else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    ap.add_argument("--ecospace", default="thesis/convergence/output_v3_fullpbdb/ecospace_genus_mapping.csv")
    ap.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    ap.add_argument("--out", default="thesis/synthesis/output_terrestrial_pilot")
    ap.add_argument("--time-bin-myr", type=float, default=10.0)
    ap.add_argument("--grid-deg", type=float, default=15.0)  # coarser for sparser terrestrial record
    ap.add_argument("--min-genera-per-region", type=int, default=10)
    ap.add_argument("--max-pairs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # Load data.
    occ = pd.read_parquet(
        args.pbdb,
        columns=["source_db", "occurrence_id", "mid_ma", "lat", "lng", "paleolat", "paleolng",
                  "phylum", "class", "order", "family", "genus"],
    )
    occ = occ[occ["source_db"] == "PBDB"].copy()
    occ["genus"] = occ["genus"].map(_clean)
    occ = occ.dropna(subset=["genus", "mid_ma"]).copy()
    occ["analysis_lat"] = occ["paleolat"].where(occ["paleolat"].notna(), occ["lat"])
    occ["analysis_lng"] = occ["paleolng"].where(occ["paleolng"].notna(), occ["lng"])
    occ = occ.dropna(subset=["analysis_lat", "analysis_lng"]).copy()

    eco = pd.read_csv(args.ecospace)
    eco["genus"] = eco["genus"].map(_clean)
    eco = eco.dropna(subset=["genus"]).copy()

    # Filter to TERRESTRIAL genera via jev field.
    eco["is_terrestrial"] = eco["jev"].astype(str).str.contains("terrestrial", case=False, na=False)
    eco_terr = eco[eco["is_terrestrial"]].copy()
    eco_terr = eco_terr.dropna(subset=["role_id"]).copy()
    print(f"Terrestrial genera with complete ecospace roles: {len(eco_terr)}")

    if len(eco_terr) < 50:
        print("Too few terrestrial genera with ecospace annotations. Trying broader 'non-marine' filter.")
        eco_terr = eco[~eco["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()
        eco_terr = eco_terr[eco_terr["jev"].notna()].copy()
        eco_terr = eco_terr.dropna(subset=["role_id"]).copy()
        print(f"Non-marine genera with complete roles: {len(eco_terr)}")

    terr_genera = set(eco_terr["genus"].unique())
    occ_terr = occ[occ["genus"].isin(terr_genera)].copy()
    print(f"Terrestrial occurrences: {len(occ_terr)}")

    # Bin and grid.
    occ_terr["time_bin"] = (pd.to_numeric(occ_terr["mid_ma"], errors="coerce") / args.time_bin_myr).round() * args.time_bin_myr
    occ_terr["lat_bin"] = (occ_terr["analysis_lat"] / args.grid_deg).round() * args.grid_deg
    occ_terr["lng_bin"] = (occ_terr["analysis_lng"] / args.grid_deg).round() * args.grid_deg
    occ_terr["locality"] = list(zip(occ_terr["lat_bin"], occ_terr["lng_bin"]))

    # Attach roles.
    occ_terr = occ_terr.merge(eco_terr[["genus", "role_id"]], on="genus", how="left")
    occ_terr = occ_terr.dropna(subset=["role_id"]).copy()
    occ_terr = occ_terr.drop_duplicates(subset=["time_bin", "locality", "genus"]).copy()

    # Build role vectors.
    role_counts = occ_terr.groupby(["time_bin", "locality", "role_id"])["genus"].nunique().rename("cnt").reset_index()
    genus_sets = occ_terr.groupby(["time_bin", "locality"])["genus"].agg(lambda s: set(s)).rename("gset").reset_index()
    genus_n = occ_terr.groupby(["time_bin", "locality"])["genus"].nunique().rename("ng").reset_index()
    genus_sets = genus_sets.merge(genus_n, on=["time_bin", "locality"])

    all_roles = sorted(role_counts["role_id"].unique())
    role_idx = {r: i for i, r in enumerate(all_roles)}

    vectors = {}
    for (t, loc), sub in role_counts.groupby(["time_bin", "locality"]):
        vec = np.zeros(len(all_roles))
        for _, row in sub.iterrows():
            vec[role_idx[row["role_id"]]] = float(row["cnt"])
        vectors[(float(t), tuple(loc))] = vec

    rng = np.random.default_rng(args.seed)
    rows = []

    for tb in sorted(occ_terr["time_bin"].unique(), reverse=True):
        locs = genus_sets[(genus_sets["time_bin"] == tb) & (genus_sets["ng"] >= args.min_genera_per_region)]
        localities = list(locs["locality"])
        n_loc = len(localities)
        if n_loc < 4:
            continue

        loc_gsets = {tuple(r["locality"]): r["gset"] for _, r in locs.iterrows()}

        pairs = [(tuple(localities[a]), tuple(localities[b]))
                 for a in range(n_loc - 1) for b in range(a + 1, n_loc)]
        if len(pairs) > args.max_pairs:
            idx = rng.choice(len(pairs), size=args.max_pairs, replace=False)
            pairs = [pairs[j] for j in idx]

        func_sims, tax_sims = [], []
        for la, lb in pairs:
            va, vb = vectors.get((float(tb), la)), vectors.get((float(tb), lb))
            if va is None or vb is None:
                continue
            f = _js_sim(va, vb)
            t_sim = _jaccard(loc_gsets.get(la, set()), loc_gsets.get(lb, set()))
            if np.isfinite(f) and np.isfinite(t_sim):
                func_sims.append(f)
                tax_sims.append(t_sim)

        if len(func_sims) < 10:
            continue

        x_arr = np.array(tax_sims)
        y_arr = np.array(func_sims)
        A = np.column_stack([np.ones(len(x_arr)), x_arr])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        resid = y_arr - A.dot(coef)

        rows.append({
            "time_bin": float(tb),
            "n_localities": n_loc,
            "n_pairs": len(func_sims),
            "mean_func_sim": float(np.mean(func_sims)),
            "mean_tax_sim": float(np.mean(tax_sims)),
            "functional_excess_similarity_js": float(np.mean(resid)),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "terrestrial_convergence_metrics.csv", index=False)

    # Merge with earth system.
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})
    merged = metrics.merge(earth, on="time_bin", how="left")
    merged = merged.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    merged = merged.sort_values("time_bin", ascending=False).reset_index(drop=True)

    results: dict[str, object] = {
        "n_terrestrial_genera": len(terr_genera),
        "n_terrestrial_occurrences": len(occ_terr),
        "n_bins": len(merged),
        "n_unique_roles": len(all_roles),
    }

    if len(merged) >= 6:
        v = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
        y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
        mask = np.isfinite(v) & np.isfinite(y)
        r = float(np.corrcoef(v[mask], y[mask])[0, 1])
        n = int(mask.sum())
        more = sum(1 for _ in range(10000) if abs(float(np.corrcoef(v[mask], np.random.default_rng(42 + _).permutation(y[mask]))[0, 1])) >= abs(r))
        p = (more + 1) / 10001
        results["volatility_vs_convergence"] = {"corr": r, "perm_p": p, "n": n}
    else:
        results["volatility_vs_convergence"] = {"note": f"too few bins ({len(merged)})"}

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Figure.
    if len(merged) >= 4:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(
            merged["delta_from_prev_T_field_meanabs"],
            merged["functional_excess_similarity_js"],
            s=50, alpha=0.8, color="#ff7f0e",
        )
        ax.set_xlabel("Climate volatility")
        ax.set_ylabel("Functional excess similarity (JS)")
        r = results.get("volatility_vs_convergence", {})
        ax.set_title(f"Terrestrial convergence pilot\nn={r.get('n', len(merged))}, r={r.get('corr', 'nan'):.3f}")
        ax.axhline(0, color="grey", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "terrestrial_pilot.png", dpi=220)
        plt.close(fig)

    # Summary.
    lines = [
        "# Terrestrial convergence pilot",
        "",
        f"- Terrestrial genera with complete roles: {len(eco_terr)}",
        f"- Terrestrial occurrences used: {len(occ_terr)}",
        f"- Unique functional roles: {len(all_roles)}",
        f"- Time bins with sufficient data: {len(merged)}",
        "",
    ]
    r = results.get("volatility_vs_convergence", {})
    if "corr" in r:
        lines.append(f"Volatility vs convergence: corr={r['corr']:.3f}, perm-p={r['perm_p']:.3g}")
    else:
        lines.append(f"Result: {r.get('note', 'N/A')}")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
