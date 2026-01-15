from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis.geographic_portfolio.run_event_portfolio_analysis import (  # noqa: E402
    DEFAULT_EVENTS,
    build_event_dataset,
    fit_repeated_logit,
)
from thesis.paleobiotic_velocity.run_pipeline import load_occurrences  # noqa: E402


CoordsMode = Literal["paleo", "modern"]


@dataclass(frozen=True)
class Hypothesis:
    name: str
    feature: str
    description: str


HYPOTHESES: list[Hypothesis] = [
    Hypothesis(
        name="portfolio_entropy",
        feature="component_entropy",
        description="More even multi-province portfolio (higher component entropy) increases survivorship beyond range size.",
    ),
    Hypothesis(
        name="equator_crossing",
        feature="cross_equator",
        description="Pre-event distributions spanning both hemispheres have higher survivorship (climate-belt buffering).",
    ),
    Hypothesis(
        name="latitudinal_position",
        feature="centroid_abs_lat",
        description="Pre-event centroid absolute latitude predicts survivorship (high-lat refugia or tropics-as-buffer).",
    ),
    Hypothesis(
        name="spatial_dispersion",
        feature="log_dispersion_km",
        description="More spatially dispersed distributions (mean distance to centroid) increase survivorship beyond range size.",
    ),
    Hypothesis(
        name="longitudinal_span",
        feature="lon_span_deg",
        description="Wider longitudinal coverage (circular span) increases survivorship by spanning provinces/basins.",
    ),
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _summarize_feature(coef_summary: pd.DataFrame, feature: str) -> dict[str, float]:
    coef = coef_summary.set_index("feature")
    if feature not in coef.index:
        return {"or": float("nan"), "or_p2_5": float("nan"), "or_p97_5": float("nan")}
    row = coef.loc[feature]
    return {
        "or": float(row["odds_ratio_mean"]),
        "or_p2_5": float(row["odds_ratio_p2_5"]),
        "or_p97_5": float(row["odds_ratio_p97_5"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--out", default="thesis/geographic_portfolio/output_additional_hypotheses")
    p.add_argument("--coords-mode", choices=["paleo", "modern", "both"], default="both")
    p.add_argument("--grid-deg", default="5,10", help="Comma-separated list (e.g., '5,10').")
    p.add_argument("--time-bin-myr", type=float, default=5.0)
    p.add_argument("--post-window-myr", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-phylum", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    grid_vals = [float(x.strip()) for x in str(args.grid_deg).split(",") if x.strip()]
    if not grid_vals:
        raise ValueError("--grid-deg must include at least one value")

    modes: list[CoordsMode]
    if args.coords_mode == "both":
        modes = ["paleo", "modern"]
    else:
        modes = [args.coords_mode]  # type: ignore[assignment]

    baseline_numeric = [
        "log_abundance",
        "log_geographic_range",
        "lat_range",
        "log_env_breadth",
        "largest_component_frac",  # prior portfolio result; keep as baseline.
    ]
    categorical = ["phylum"] if args.with_phylum else []

    summary_rows: list[dict] = []

    for mode in modes:
        for grid_deg in grid_vals:
            print(f"Loading occurrences (coords_mode={mode}, grid_deg={grid_deg})…")
            occ = load_occurrences(
                args.data,
                coords_mode=mode,  # type: ignore[arg-type]
                time_bin_myr=float(args.time_bin_myr),
                locality_bin_deg=float(grid_deg),
            )

            combo_dir = out_dir / mode / f"grid_{int(grid_deg)}deg"
            _ensure_dir(combo_dir / "datasets")
            _ensure_dir(combo_dir / "results")

            for ev in DEFAULT_EVENTS:
                ds = build_event_dataset(
                    occ,
                    boundary_ma=ev.boundary_ma,
                    time_bin_myr=float(args.time_bin_myr),
                    grid_deg=float(grid_deg),
                    post_window_myr=float(args.post_window_myr),
                )

                for target in ["survived_any", "survived_10myr"]:
                    dataset_path = combo_dir / "datasets" / f"{ev.name}_{target}_dataset.parquet"
                    ds.to_parquet(dataset_path, index=False)

                    # Cache baseline fit for this dataset/target.
                    base_fit = fit_repeated_logit(
                        ds,
                        target=target,
                        numeric_features=baseline_numeric,
                        categorical_features=categorical,
                        repeats=int(args.repeats),
                        seed=int(args.seed),
                    )

                    (combo_dir / "results" / f"{ev.name}_{target}_baseline_coef_summary.csv").write_text(
                        base_fit["coef_summary"].to_csv(index=False)
                    )
                    (combo_dir / "results" / f"{ev.name}_{target}_baseline_repeats.csv").write_text(
                        base_fit["repeats"].to_csv(index=False)
                    )

                    for hyp in HYPOTHESES:
                        full_numeric = baseline_numeric + [hyp.feature]
                        full_fit = fit_repeated_logit(
                            ds,
                            target=target,
                            numeric_features=full_numeric,
                            categorical_features=categorical,
                            repeats=int(args.repeats),
                            seed=int(args.seed),
                        )

                        feat = _summarize_feature(full_fit["coef_summary"], hyp.feature)
                        delta_auc = float(full_fit["summary"]["auc_mean"] - base_fit["summary"]["auc_mean"])

                        meta = {
                            "coords_mode": mode,
                            "grid_deg": float(grid_deg),
                            "event": ev.__dict__,
                            "target": target,
                            "hypothesis": hyp.__dict__,
                            "time_bin_myr": float(args.time_bin_myr),
                            "post_window_myr": float(args.post_window_myr),
                            "with_phylum": bool(args.with_phylum),
                            "full_model": full_fit["summary"],
                            "baseline_model": base_fit["summary"],
                            "delta_auc_full_minus_baseline": delta_auc,
                            "feature_odds_ratio": feat,
                        }

                        out_prefix = f"{ev.name}_{target}_{hyp.name}"
                        (combo_dir / "results" / f"{out_prefix}_coef_summary.csv").write_text(
                            full_fit["coef_summary"].to_csv(index=False)
                        )
                        (combo_dir / "results" / f"{out_prefix}_repeats.csv").write_text(
                            full_fit["repeats"].to_csv(index=False)
                        )
                        (combo_dir / "results" / f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

                        summary_rows.append(
                            {
                                "coords_mode": mode,
                                "grid_deg": float(grid_deg),
                                "event": ev.name,
                                "boundary_ma": ev.boundary_ma,
                                "target": target,
                                "hypothesis": hyp.name,
                                "feature": hyp.feature,
                                "n": full_fit["summary"]["n_rows"],
                                "event_rate": full_fit["summary"]["event_rate"],
                                "auc_full": full_fit["summary"]["auc_mean"],
                                "auc_base": base_fit["summary"]["auc_mean"],
                                "delta_auc": delta_auc,
                                "or_feature": feat["or"],
                                "or_feature_p2_5": feat["or_p2_5"],
                                "or_feature_p97_5": feat["or_p97_5"],
                            }
                        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

