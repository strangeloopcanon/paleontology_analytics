"""Hierarchical model for geographic portfolio across all mass-extinction events.

Replaces per-event logistic regressions with a single hierarchical model where
event is a random intercept and connectedness has a fixed effect plus
event-level random slopes. Tests three operationalizations of "configuration":
largest_component_frac, range_entropy, dispersion_km.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_MIXEDLM = True
except ImportError:
    HAS_MIXEDLM = False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _z(x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 3:
        return out
    mu, sd = float(np.mean(x[mask])), float(np.std(x[mask], ddof=1))
    sd = sd if sd > 0 else 1.0
    out[mask] = (x[mask] - mu) / sd
    return out


DEFAULT_EVENTS = {
    "end_ordovician": 444.0,
    "late_devonian": 372.0,
    "end_permian": 252.0,
    "end_triassic": 201.0,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default="thesis/geographic_portfolio/output",
        help="Directory containing per-event CSVs from run_event_portfolio_analysis.py"
    )
    ap.add_argument("--out", default="thesis/geographic_portfolio/output_hierarchical")
    args = ap.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    if not HAS_MIXEDLM:
        msg = "statsmodels required for mixed-effects model"
        (out_dir / "analysis_results.json").write_text(json.dumps({"error": msg}) + "\n")
        print(msg)
        return

    # Load per-event data.
    dfs = []
    data_dir = Path(args.data_dir)
    for event_name, boundary_ma in DEFAULT_EVENTS.items():
        csv_path = data_dir / f"{event_name}_genera.csv"
        if not csv_path.exists():
            csv_path = data_dir / f"genera_{event_name}.csv"
        if not csv_path.exists():
            for p in data_dir.glob(f"*{event_name}*genera*.csv"):
                csv_path = p
                break
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["event"] = event_name
            df["boundary_ma"] = boundary_ma
            dfs.append(df)

    if not dfs:
        msg = f"No per-event CSVs found in {data_dir}"
        (out_dir / "analysis_results.json").write_text(json.dumps({"error": msg}) + "\n")
        print(msg)
        return

    combined = pd.concat(dfs, ignore_index=True)

    config_cols = ["largest_component_frac"]
    for col in ["range_entropy", "component_entropy", "dispersion_km"]:
        if col in combined.columns:
            config_cols.append(col)

    outcome_col = None
    for candidate in ["survived_10myr", "survived_any", "survived"]:
        if candidate in combined.columns:
            outcome_col = candidate
            break

    if outcome_col is None:
        msg = "No survivorship column found"
        (out_dir / "analysis_results.json").write_text(json.dumps({"error": msg}) + "\n")
        print(msg)
        return

    results: dict[str, object] = {
        "n_events": len(dfs),
        "events": list(DEFAULT_EVENTS.keys()),
        "n_genera_total": len(combined),
        "outcome": outcome_col,
    }

    for config_col in config_cols:
        if config_col not in combined.columns:
            results[f"model_{config_col}"] = {"error": f"column {config_col} not found"}
            continue

        sub = combined.dropna(subset=[config_col, outcome_col]).copy()
        if len(sub) < 20:
            results[f"model_{config_col}"] = {"error": f"too few observations ({len(sub)})"}
            continue

        sub[f"{config_col}_z"] = _z(sub[config_col].to_numpy(dtype=float))

        # Control covariates (z-scored).
        control_cols_z = []
        for raw_col in ["log_geographic_range", "log_n_occurrences", "lat_range"]:
            if raw_col in sub.columns:
                sub[f"{raw_col}_z"] = _z(sub[raw_col].to_numpy(dtype=float))
                control_cols_z.append(f"{raw_col}_z")

        feature_col = f"{config_col}_z"
        all_x_cols = [feature_col] + control_cols_z
        sub = sub.dropna(subset=all_x_cols + [outcome_col, "event"]).copy()

        if len(sub) < 20:
            results[f"model_{config_col}"] = {"error": f"too few obs after dropna ({len(sub)})"}
            continue

        y = sub[outcome_col].astype(float).to_numpy()
        X = sub[all_x_cols].astype(float)
        X = sm.add_constant(X)
        groups = sub["event"]

        try:
            # Random intercept + random slope for the configuration variable.
            model = MixedLM(
                endog=y,
                exog=X,
                groups=groups,
                exog_re=sub[[feature_col]].astype(float),
            )
            fit = model.fit(reml=True)
            fe_params = dict(zip(X.columns, fit.fe_params))
            fe_pvalues = dict(zip(X.columns, fit.pvalues[:len(X.columns)]))

            results[f"model_{config_col}"] = {
                "n_obs": len(sub),
                "n_events": int(groups.nunique()),
                "fixed_effect_beta": fe_params.get(feature_col, float("nan")),
                "fixed_effect_p": fe_pvalues.get(feature_col, float("nan")),
                "all_fixed_effects": {k: {"beta": float(v), "p": float(fe_pvalues.get(k, float("nan")))} for k, v in fe_params.items()},
                "converged": fit.converged,
                "log_likelihood": float(fit.llf),
            }
        except Exception as exc:
            # Fallback to random intercept only.
            try:
                model_ri = MixedLM(endog=y, exog=X, groups=groups)
                fit_ri = model_ri.fit(reml=True)
                fe_params = dict(zip(X.columns, fit_ri.fe_params))
                fe_pvalues = dict(zip(X.columns, fit_ri.pvalues[:len(X.columns)]))
                results[f"model_{config_col}"] = {
                    "n_obs": len(sub),
                    "n_events": int(groups.nunique()),
                    "fixed_effect_beta": fe_params.get(feature_col, float("nan")),
                    "fixed_effect_p": fe_pvalues.get(feature_col, float("nan")),
                    "note": f"random-intercept only (random slope failed: {exc})",
                    "converged": fit_ri.converged,
                }
            except Exception as exc2:
                results[f"model_{config_col}"] = {"error": str(exc2)}

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    lines = [
        "# Hierarchical model: geographic portfolio across events",
        "",
        f"- Events: {', '.join(results['events'])}",
        f"- Total genera: {results['n_genera_total']}",
        f"- Outcome: {results['outcome']}",
        "",
    ]
    for config_col in config_cols:
        r = results.get(f"model_{config_col}", {})
        lines.append(f"## {config_col}")
        if "error" in r:
            lines.append(f"- {r['error']}")
        else:
            lines.append(f"- n = {r.get('n_obs', '?')}, events = {r.get('n_events', '?')}")
            lines.append(f"- Fixed effect (z-scored): beta = {r.get('fixed_effect_beta', 'nan'):.4f}, p = {r.get('fixed_effect_p', 'nan'):.3g}")
            if "note" in r:
                lines.append(f"- Note: {r['note']}")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
