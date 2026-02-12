"""Test whether the convergence signal survives restricting to well-annotated clades.

Reruns the core convergence metric computation for subsets:
- Bivalvia only
- Gastropoda only
- Brachiopoda only
- All three combined ("well-annotated marine")

Then tests volatility association for each subset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    p = a / a.sum()
    q = b / b.sum()
    d = float(jensenshannon(p, q))
    return float(1.0 - np.clip(d, 0, 1)) if np.isfinite(d) else float("nan")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    u = len(a | b)
    return float(len(a & b) / u) if u else float("nan")


def _partial_corr_shift(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if int(np.sum(mask)) < 8:
        return {"corr": float("nan"), "p_shift": float("nan"), "n": int(np.sum(mask))}
    xm, ym, cm = x[mask], y[mask], controls[mask]
    A = np.column_stack([np.ones(len(cm)), cm])
    bx, *_ = np.linalg.lstsq(A, xm, rcond=None)
    by, *_ = np.linalg.lstsq(A, ym, rcond=None)
    rx, ry = xm - A.dot(bx), ym - A.dot(by)
    obs = float(np.corrcoef(rx, ry)[0, 1])
    n = len(rx)
    more = sum(1 for s in range(n) if abs(float(np.corrcoef(rx, np.roll(ry, s))[0, 1])) >= abs(obs))
    return {"corr": obs, "p_shift": more / n, "n": n}


def compute_convergence_for_subset(
    occ: pd.DataFrame,
    eco_map: pd.DataFrame,
    *,
    time_bin_myr: float,
    grid_deg: float,
    min_genera_per_region: int,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """Compute per-bin functional excess similarity for a filtered occurrence set."""
    df = occ.copy()
    df["time_bin"] = (pd.to_numeric(df["mid_ma"], errors="coerce") / time_bin_myr).round() * time_bin_myr
    df["lat_bin"] = (df["analysis_lat"] / grid_deg).round() * grid_deg
    df["lng_bin"] = (df["analysis_lng"] / grid_deg).round() * grid_deg
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))

    # Attach roles.
    df = df.merge(eco_map[["genus", "role_id"]], on="genus", how="left")
    df = df.dropna(subset=["role_id"]).copy()
    df = df.drop_duplicates(subset=["time_bin", "locality", "genus"]).copy()

    # Build role vectors and genus sets.
    role_counts = df.groupby(["time_bin", "locality", "role_id"])["genus"].nunique().rename("cnt").reset_index()
    genus_sets = df.groupby(["time_bin", "locality"])["genus"].agg(lambda s: set(s)).rename("gset").reset_index()
    genus_n = df.groupby(["time_bin", "locality"])["genus"].nunique().rename("ng").reset_index()
    genus_sets = genus_sets.merge(genus_n, on=["time_bin", "locality"])

    all_roles = sorted(role_counts["role_id"].unique())
    role_idx = {r: i for i, r in enumerate(all_roles)}

    vectors = {}
    for (t, loc), sub in role_counts.groupby(["time_bin", "locality"]):
        vec = np.zeros(len(all_roles))
        for _, row in sub.iterrows():
            vec[role_idx[row["role_id"]]] = float(row["cnt"])
        vectors[(float(t), tuple(loc))] = vec

    rng = np.random.default_rng(seed)
    rows = []

    for tb in sorted(df["time_bin"].unique(), reverse=True):
        locs = genus_sets[(genus_sets["time_bin"] == tb) & (genus_sets["ng"] >= min_genera_per_region)]
        localities = list(locs["locality"])
        n_loc = len(localities)
        if n_loc < 6:
            continue

        loc_gsets = {tuple(r["locality"]): r["gset"] for _, r in locs.iterrows()}

        pairs = [(tuple(localities[a]), tuple(localities[b])) for a in range(n_loc - 1) for b in range(a + 1, n_loc)]
        if len(pairs) > max_pairs:
            idx = rng.choice(len(pairs), size=max_pairs, replace=False)
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

        if len(func_sims) < 100:
            continue

        # Global fit for residual.
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
            "provinciality": float(1.0 - np.mean(tax_sims)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbdb", default="data/processed/merged_occurrences.parquet")
    ap.add_argument("--ecospace", default="thesis/convergence/output_v3_fullpbdb/ecospace_genus_mapping.csv")
    ap.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    ap.add_argument("--out", default="thesis/synthesis/output_clade_restriction")
    ap.add_argument("--time-bin-myr", type=float, default=10.0)
    ap.add_argument("--grid-deg", type=float, default=10.0)
    ap.add_argument("--min-genera-per-region", type=int, default=15)
    ap.add_argument("--max-pairs", type=int, default=20000)
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

    # Paleocoordinates.
    occ["analysis_lat"] = occ["paleolat"].where(occ["paleolat"].notna(), occ["lat"])
    occ["analysis_lng"] = occ["paleolng"].where(occ["paleolng"].notna(), occ["lng"])
    occ = occ.dropna(subset=["analysis_lat", "analysis_lng"]).copy()

    eco = pd.read_csv(args.ecospace)
    eco["genus"] = eco["genus"].map(_clean)
    eco = eco.dropna(subset=["genus"]).copy()
    # Filter to marine.
    eco = eco[eco["jev"].astype(str).str.contains("marine", case=False, na=False)].copy()
    eco = eco.dropna(subset=["role_id"]).copy()

    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"})

    # Define clade subsets.
    CLADES = {
        "Bivalvia": {"class": ["Bivalvia"]},
        "Gastropoda": {"class": ["Gastropoda"]},
        "Brachiopoda": {"phylum": ["Brachiopoda"]},
        "well_annotated_combined": {
            "class": ["Bivalvia", "Gastropoda"],
            "phylum": ["Brachiopoda"],
        },
    }

    results = {}
    all_metrics = {}

    for clade_name, filters in CLADES.items():
        # Filter occurrences.
        mask = pd.Series(False, index=occ.index)
        for col, vals in filters.items():
            mask |= occ[col].isin(vals)
        occ_sub = occ[mask].copy()

        # Also restrict ecospace to genera in this subset.
        genera_in_sub = set(occ_sub["genus"].unique())
        eco_sub = eco[eco["genus"].isin(genera_in_sub)].copy()

        if len(eco_sub) < 50:
            results[clade_name] = {"error": f"too few genera with ecospace ({len(eco_sub)})"}
            continue

        metrics = compute_convergence_for_subset(
            occ_sub, eco_sub,
            time_bin_myr=args.time_bin_myr, grid_deg=args.grid_deg,
            min_genera_per_region=args.min_genera_per_region,
            max_pairs=args.max_pairs, seed=args.seed,
        )

        if len(metrics) < 8:
            results[clade_name] = {"error": f"too few bins ({len(metrics)})"}
            continue

        # Merge with earth system.
        merged = metrics.merge(earth, on="time_bin", how="left")
        merged = merged.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
        merged = merged.sort_values("time_bin", ascending=False).reset_index(drop=True)

        if len(merged) < 8:
            results[clade_name] = {"error": f"too few merged bins ({len(merged)})"}
            continue

        y = merged["functional_excess_similarity_js"].to_numpy(dtype=float)
        v = merged["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
        t = merged["time_bin"].to_numpy(dtype=float)

        test = _partial_corr_shift(v, y, t.reshape(-1, 1))
        results[clade_name] = {
            "n_bins": len(merged),
            "n_genera_with_ecospace": len(eco_sub),
            "volatility_vs_convergence_partial_time": test,
        }
        all_metrics[clade_name] = merged
        merged.to_csv(out_dir / f"metrics_{clade_name}.csv", index=False)

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Figure: clade comparison.
    fig, axes = plt.subplots(1, min(4, len(all_metrics)), figsize=(4.5 * min(4, len(all_metrics)), 4.5), squeeze=False)
    for i, (clade_name, merged) in enumerate(all_metrics.items()):
        if i >= 4:
            break
        ax = axes[0][i]
        ax.scatter(
            merged["delta_from_prev_T_field_meanabs"],
            merged["functional_excess_similarity_js"],
            s=40, alpha=0.8, color="#1f77b4",
        )
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Excess similarity")
        r = results[clade_name].get("volatility_vs_convergence_partial_time", {})
        ax.set_title(f"{clade_name}\nr={r.get('corr', 'nan'):.2f}, p={r.get('p_shift', 'nan'):.3f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "clade_restriction_comparison.png", dpi=220)
    plt.close(fig)

    # Summary.
    lines = ["# Clade restriction test", ""]
    for clade_name, res in results.items():
        if "error" in res:
            lines.append(f"- {clade_name}: {res['error']}")
        else:
            test = res.get("volatility_vs_convergence_partial_time", {})
            lines.append(
                f"- {clade_name} (n_bins={res['n_bins']}, n_genera={res['n_genera_with_ecospace']}): "
                f"corr={test.get('corr', 'nan'):.3f}, shift-p={test.get('p_shift', 'nan'):.3g}"
            )
    lines.extend(["", "## Files", f"- Stats: `{out_dir / 'analysis_results.json'}`", f"- Figures: `{fig_dir}`"])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
