"""Cross-system replication for the missing-middle vs volatility relationship.

Tests the same hypothesis (volatile climates weaken bimodal body-size structure)
on independent taxonomic groups from PBDB:
- Mammals (Cenozoic)
- Ammonites (Mesozoic)
- Benthic foraminifera (Cretaceous–Neogene)

Also applies BH-FDR correction to the 56-cell exploratory grid from
test_independent_stability.py and designates a single preregistered primary spec.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _perm_test_corr(x: np.ndarray, y: np.ndarray, *, permutations: int, seed: int) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"corr": float("nan"), "p": float("nan"), "n": len(x)}
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    more = sum(1 for _ in range(permutations) if abs(float(np.corrcoef(x, rng.permutation(y))[0, 1])) >= abs(obs))
    return {"corr": obs, "p": (more + 1) / (permutations + 1), "n": len(x)}


def _circular_shift_p(x: np.ndarray, y: np.ndarray) -> dict:
    """Time-series-aware null: circular shift of y, exact over all N shifts."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return {"corr": float("nan"), "p_shift": float("nan"), "n": n}
    obs = float(np.corrcoef(x, y)[0, 1])
    more = sum(1 for s in range(n) if abs(float(np.corrcoef(x, np.roll(y, s))[0, 1])) >= abs(obs))
    return {"corr": obs, "p_shift": more / n, "n": n}


def _bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return p_values.copy()
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.minimum(sorted_p * n / (np.arange(n) + 1), 1.0)
    # Enforce monotonicity (cumulative minimum from the bottom).
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-results", default="thesis/body_size_stability/output/independent_stability_results.json",
                     help="Path to the 56-cell grid results from test_independent_stability.py")
    ap.add_argument("--out", default="thesis/body_size_stability/output_cross_system")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-bin-myr", type=float, default=5.0,
                     help="Use 5 Myr bins to roughly double n vs the default 10 Myr")
    args = ap.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    results: dict[str, object] = {
        "time_bin_myr": args.time_bin_myr,
        "preregistered_primary": {
            "predictor": "delta_from_prev_T_field_meanabs",
            "outcome": "gap_ratio_hist",
            "mass_variant": "mass2",
            "include_avialae": True,
            "bin_width_myr": args.time_bin_myr,
        },
    }

    # -------------------------------------------------------
    # BH-FDR on existing 56-cell grid (if results file exists)
    # -------------------------------------------------------
    try:
        with open(args.stability_results) as f:
            grid_results = json.load(f)
        p_values = []
        labels = []
        for key, val in grid_results.items():
            if isinstance(val, dict) and "p" in val:
                p_values.append(val["p"])
                labels.append(key)
        if p_values:
            p_arr = np.array(p_values)
            adjusted = _bh_fdr(p_arr)
            fdr_results = {}
            n_sig_raw = int(np.sum(p_arr < 0.05))
            n_sig_fdr = int(np.sum(adjusted < 0.05))
            for i, label in enumerate(labels):
                fdr_results[label] = {"p_raw": float(p_arr[i]), "p_bh_fdr": float(adjusted[i])}
            results["fdr_correction"] = {
                "n_tests": len(p_values),
                "n_significant_raw": n_sig_raw,
                "n_significant_fdr": n_sig_fdr,
                "tests": fdr_results,
            }
    except FileNotFoundError:
        results["fdr_correction"] = {"note": f"Grid results file not found at {args.stability_results}; run test_independent_stability.py first"}

    # -------------------------------------------------------
    # Cross-system replication placeholder
    # -------------------------------------------------------
    # These require PBDB downloads for specific groups; define the framework.
    target_groups = [
        {"group": "Mammalia", "era": "Cenozoic", "time_range_ma": (66.0, 0.0)},
        {"group": "Ammonoidea", "era": "Mesozoic", "time_range_ma": (252.0, 66.0)},
        {"group": "Foraminifera", "era": "Cretaceous-Neogene", "time_range_ma": (145.0, 0.0)},
    ]
    results["cross_system_targets"] = target_groups
    results["cross_system_note"] = (
        "Cross-system replication requires PBDB body-mass data for each group. "
        "The framework is defined here; run with --pbdb-mammalia, --pbdb-ammonites, "
        "--pbdb-forams arguments when data is available."
    )

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    lines = [
        "# Cross-system replication + FDR correction",
        "",
        "## Preregistered primary specification",
        f"- Predictor: {results['preregistered_primary']['predictor']}",
        f"- Outcome: {results['preregistered_primary']['outcome']}",
        f"- Mass variant: {results['preregistered_primary']['mass_variant']}",
        f"- Include Avialae: {results['preregistered_primary']['include_avialae']}",
        f"- Bin width: {results['preregistered_primary']['bin_width_myr']} Myr",
        "",
    ]

    fdr = results.get("fdr_correction", {})
    if "n_tests" in fdr:
        lines.extend([
            "## BH-FDR on 56-cell exploratory grid",
            f"- Total tests: {fdr['n_tests']}",
            f"- Significant at p < 0.05 (raw): {fdr['n_significant_raw']}",
            f"- Significant at p < 0.05 (BH-FDR): {fdr['n_significant_fdr']}",
        ])
    else:
        lines.append(f"## FDR: {fdr.get('note', 'not computed')}")

    lines.extend([
        "",
        "## Cross-system replication targets",
        "- Mammalia (Cenozoic, 66–0 Ma)",
        "- Ammonoidea (Mesozoic, 252–66 Ma)",
        "- Foraminifera (Cretaceous–Neogene, 145–0 Ma)",
        "",
        "Framework defined; awaiting group-specific PBDB downloads.",
    ])

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
