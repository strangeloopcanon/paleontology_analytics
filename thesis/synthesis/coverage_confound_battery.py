"""Three orthogonal attacks on the coverage confound.

(i)   Errors-in-variables / Deming regression treating frac_marine_with_role
      as a noisy proxy for the latent true role distribution.
(ii)  Restrict to genera in the top-quartile of ecospace annotation completeness
      and rerun the headline partial correlation.
(iii) Decompose coverage into linear, quadratic, and time-detrended components
      and partial out each separately.

Reads the same merged bins table as robustness_battery.py, plus the ecospace
coverage-per-bin CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis._lib import build_controls, ensure_dir, partial_corr, residualize


def _shift_p_residualized(rv: np.ndarray, ry: np.ndarray) -> dict:
    """Circular-shift p-value on pre-residualized arrays."""
    mask = np.isfinite(rv) & np.isfinite(ry)
    rv, ry = rv[mask], ry[mask]
    n = len(rv)
    if n < 6:
        return {"corr": float("nan"), "p_shift": float("nan"), "n": n}
    obs = float(np.corrcoef(rv, ry)[0, 1])
    more = sum(1 for s in range(n) if abs(float(np.corrcoef(rv, np.roll(ry, s))[0, 1])) >= abs(obs))
    return {"corr": obs, "p_shift": more / n, "n": n}


def _deming_regression(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict:
    """Total least-squares (Deming) regression of y on x with error-variance ratio lambda.

    lambda = var(error_x) / var(error_y). lambda=1 assumes equal measurement
    noise in both variables — a common default when the true error ratio is
    unknown. Results are sensitive to lambda; interpret as an approximate
    correction, not an exact solution.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    n = len(xm)
    if n < 4:
        return {"slope": float("nan"), "intercept": float("nan"), "n": n}
    sxx = float(np.var(xm, ddof=1))
    syy = float(np.var(ym, ddof=1))
    sxy = float(np.cov(xm, ym, ddof=1)[0, 1])
    b1 = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy) if abs(sxy) > 1e-12 else float("nan")
    b0 = float(np.mean(ym) - b1 * np.mean(xm))
    return {"slope": b1, "intercept": b0, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument(
        "--coverage",
        default="thesis/synthesis/output_ecospace_missingness/ecospace_coverage_per_bin.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_coverage_confound_battery")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    y = bins["functional_excess_similarity_js"].to_numpy(dtype=float)
    v = bins["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    controls_base = build_controls(bins)

    results: dict[str, object] = {}

    # Baseline partial-r for reference.
    results["baseline"] = {
        "partial_r": partial_corr(v, y, controls_base),
        "controls": "time + sampling_PCA_PC12 + provinciality",
    }

    # Load coverage data.
    try:
        cov_df = pd.read_csv(args.coverage)
        cov_merged = bins.merge(cov_df[["time_bin", "frac_marine_with_role"]], on="time_bin", how="left")
        cov_col = cov_merged["frac_marine_with_role"].to_numpy(dtype=float)
    except FileNotFoundError:
        (out_dir / "analysis_results.json").write_text(
            json.dumps({"error": "coverage CSV not found"}, indent=2) + "\n"
        )
        return

    # -------------------------------------------------------
    # Attack (i): Errors-in-variables / Deming regression
    # -------------------------------------------------------
    # Treat coverage as a noisy measure of true role annotation completeness.
    # Deming regression with lambda=1 (equal noise in x and y).
    rv = residualize(v, controls_base)
    ry = residualize(y, controls_base)
    rc = residualize(cov_col, controls_base)
    mask_eiv = np.isfinite(rv) & np.isfinite(ry) & np.isfinite(rc)
    if mask_eiv.sum() >= 6:
        # Partial out coverage from both v and y using Deming regression.
        deming_v = _deming_regression(rc[mask_eiv], rv[mask_eiv], lam=1.0)
        deming_y = _deming_regression(rc[mask_eiv], ry[mask_eiv], lam=1.0)
        rv_eiv = rv[mask_eiv] - (deming_v["intercept"] + deming_v["slope"] * rc[mask_eiv])
        ry_eiv = ry[mask_eiv] - (deming_y["intercept"] + deming_y["slope"] * rc[mask_eiv])
        r_eiv = float(np.corrcoef(rv_eiv, ry_eiv)[0, 1]) if len(rv_eiv) >= 6 else float("nan")
        results["errors_in_variables"] = {
            "partial_r_deming": r_eiv,
            "n": int(mask_eiv.sum()),
            "deming_slope_v_on_cov": deming_v["slope"],
            "deming_slope_y_on_cov": deming_y["slope"],
        }
    else:
        results["errors_in_variables"] = {"error": "insufficient data"}

    # -------------------------------------------------------
    # Attack (ii): Restrict to top-quartile annotated genera
    # -------------------------------------------------------
    # Coverage restricts at the bin level: keep only bins where
    # frac_marine_with_role >= 75th percentile.
    mask_finite = np.isfinite(cov_col)
    if mask_finite.sum() >= 10:
        q75 = float(np.percentile(cov_col[mask_finite], 75))
        high_cov = cov_col >= q75
        n_high = int(high_cov.sum())
        if n_high >= 8:
            r_high = partial_corr(v[high_cov], y[high_cov], controls_base[high_cov])
            shift_rv = residualize(v[high_cov], controls_base[high_cov])
            shift_ry = residualize(y[high_cov], controls_base[high_cov])
            shift_result = _shift_p_residualized(shift_rv, shift_ry)
            results["top_quartile_coverage"] = {
                "q75_threshold": q75,
                "n_bins": n_high,
                "partial_r": r_high,
                **shift_result,
            }
        else:
            results["top_quartile_coverage"] = {"error": f"too few bins above Q75 ({n_high})"}
    else:
        results["top_quartile_coverage"] = {"error": "insufficient coverage data"}

    # -------------------------------------------------------
    # Attack (iii): Decompose coverage into components
    # -------------------------------------------------------
    t = bins["time_bin"].to_numpy(dtype=float)
    decomp_results = {}
    for label, extra_cols in [
        ("linear", [cov_col]),
        ("linear_quadratic", [cov_col, cov_col ** 2]),
        ("time_detrended", [residualize(cov_col, t.reshape(-1, 1))]),
    ]:
        extras = [c for c in extra_cols if np.isfinite(c).sum() >= 6]
        if not extras:
            decomp_results[label] = {"error": "insufficient data"}
            continue
        controls_ext = np.column_stack([controls_base] + extras)
        r_ext = partial_corr(v, y, controls_ext)
        rv_ext = residualize(v, controls_ext)
        ry_ext = residualize(y, controls_ext)
        shift_ext = _shift_p_residualized(rv_ext, ry_ext)
        decomp_results[label] = {"partial_r": r_ext, **shift_ext}
    results["coverage_decomposition"] = decomp_results

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    lines = [
        "# Coverage confound battery",
        "",
        f"## Baseline: partial r = {results['baseline']['partial_r']:.4f}",
        "",
        "## Attack (i): Errors-in-variables (Deming regression)",
    ]
    eiv = results.get("errors_in_variables", {})
    if "error" in eiv:
        lines.append(f"- {eiv['error']}")
    else:
        lines.append(f"- Partial r (Deming-corrected): {eiv['partial_r_deming']:.4f} (n={eiv['n']})")

    lines.extend(["", "## Attack (ii): Top-quartile coverage bins only"])
    tq = results.get("top_quartile_coverage", {})
    if "error" in tq:
        lines.append(f"- {tq['error']}")
    else:
        lines.append(f"- Threshold: frac_marine_with_role >= {tq['q75_threshold']:.3f}")
        lines.append(f"- n_bins = {tq['n_bins']}, partial r = {tq['partial_r']:.4f}, shift-p = {tq.get('p_shift', 'nan'):.3g}")

    lines.extend(["", "## Attack (iii): Coverage decomposition"])
    for k, v_res in results.get("coverage_decomposition", {}).items():
        if "error" in v_res:
            lines.append(f"- {k}: {v_res['error']}")
        else:
            lines.append(f"- {k}: partial r = {v_res.get('partial_r', 'nan'):.4f}, shift-p = {v_res.get('p_shift', 'nan'):.3g}")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
