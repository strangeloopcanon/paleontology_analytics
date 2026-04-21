"""Modern analog identification.

Identifies the deep-time bins whose climate volatility (|ΔT|) most resembles
current anthropogenic warming rates. Reports the convergence patterns observed
in those analog intervals as a forward-looking framing for the manuscript.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from thesis._lib import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bins",
        default="thesis/synthesis/output_sampling_autocorr/merged.csv",
    )
    ap.add_argument("--out", default="thesis/synthesis/output_modern_analog")
    ap.add_argument(
        "--modern-rate-c-per-10myr", type=float, default=4.0,
        help="Approximate peak transient |ΔT| that would be recorded in a 10 Myr "
             "CESM-like snapshot comparison spanning the anthropogenic perturbation. "
             "Default 4°C reflects ~4°C warming over 200-300 years under high-emissions, "
             "damped over the remainder of a 10 Myr window by equilibration. "
             "This is an upper-bound estimate; sensitivity to this value should be "
             "explored by varying the flag."
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    bins = pd.read_csv(args.bins)
    bins = bins.dropna(subset=["functional_excess_similarity_js", "delta_from_prev_T_field_meanabs"]).copy()
    bins = bins.sort_values("time_bin", ascending=False).reset_index(drop=True)

    vol = bins["delta_from_prev_T_field_meanabs"].to_numpy(dtype=float)
    excess = bins["functional_excess_similarity_js"].to_numpy(dtype=float)
    tb = bins["time_bin"].to_numpy(dtype=float)

    # Rank by proximity to the modern rate.
    distance = np.abs(vol - args.modern_rate_c_per_10myr)
    rank_idx = np.argsort(distance)

    n_analogs = min(5, len(bins))
    analogs = []
    for i in rank_idx[:n_analogs]:
        analogs.append({
            "time_bin_ma": float(tb[i]),
            "volatility": float(vol[i]),
            "excess_similarity": float(excess[i]),
            "distance_from_modern": float(distance[i]),
        })

    # Context: percentile of each analog's volatility.
    vol_sorted = np.sort(vol)
    for a in analogs:
        a["vol_percentile"] = float(np.searchsorted(vol_sorted, a["volatility"]) / len(vol_sorted) * 100)

    # High-volatility bins (top quartile) summary.
    q75 = float(np.percentile(vol, 75))
    high_vol_mask = vol >= q75
    high_vol_excess = excess[high_vol_mask]

    results = {
        "modern_rate_c_per_10myr": args.modern_rate_c_per_10myr,
        "phanerozoic_vol_range": [float(vol.min()), float(vol.max())],
        "phanerozoic_vol_mean": float(vol.mean()),
        "phanerozoic_vol_median": float(np.median(vol)),
        "analogs": analogs,
        "high_vol_q75_threshold": q75,
        "high_vol_n_bins": int(high_vol_mask.sum()),
        "high_vol_mean_excess": float(high_vol_excess.mean()) if len(high_vol_excess) > 0 else float("nan"),
    }

    (out_dir / "analysis_results.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# Modern analog identification",
        "",
        f"Reference rate: ~{args.modern_rate_c_per_10myr}°C global |ΔT| per 10 Myr equivalent",
        f"(Phanerozoic range: {vol.min():.2f}–{vol.max():.2f}°C, median={np.median(vol):.2f}°C)",
        "",
        "## Closest deep-time analogs",
        "",
        "| Rank | Time (Ma) | |ΔT| (°C) | Percentile | Excess similarity |",
        "|------|-----------|-----------|------------|-------------------|",
    ]
    for i, a in enumerate(analogs):
        lines.append(
            f"| {i+1} | {a['time_bin_ma']:.0f} | {a['volatility']:.2f} | "
            f"{a['vol_percentile']:.0f}th | {a['excess_similarity']:.4f} |"
        )

    lines.extend([
        "",
        "## High-volatility bins (top quartile)",
        f"- Threshold: |ΔT| >= {q75:.2f}°C ({results['high_vol_n_bins']} bins)",
        f"- Mean excess functional similarity: {results['high_vol_mean_excess']:.4f}",
        "",
        "## Interpretation",
        "If anthropogenic warming pushes climate volatility into the range of",
        "the most volatile Phanerozoic intervals, our results predict that",
        "geographically separated marine ecosystems will increasingly converge",
        "on similar ecological role mixtures — regardless of their taxonomic",
        "composition.",
    ])

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
