from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _angle_component_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose taxon velocity into components parallel/orthogonal to the global sampling shift direction.

    Uses the existing:
    - velocity_km_per_myr_back (capped)
    - align_with_global_sampling (cosine similarity proxy; in [-1, 1])
    """
    out = df.copy()
    align = out["align_with_global_sampling"].astype(float).clip(-1.0, 1.0)
    out["velocity_parallel_sampling"] = out["velocity_km_per_myr_back"] * align
    out["velocity_orth_sampling"] = out["velocity_km_per_myr_back"] * np.sqrt(np.maximum(0.0, 1.0 - align**2))
    return out


@dataclass(frozen=True)
class FitResult:
    repeats: pd.DataFrame
    summary: dict
    coef_summary: pd.DataFrame


def _fit_discrete_time_logit(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    group_col: str,
    time_col: str,
    include_time_fixed_effects: bool,
    repeats: int,
    random_state: int,
) -> FitResult:
    y = data["event_extinct_next_bin"].astype(int)
    groups = data[group_col]

    transformers: list[tuple[str, object, list[str]]] = [("num", StandardScaler(), feature_cols)]
    if include_time_fixed_effects:
        transformers.append(("time", OneHotEncoder(handle_unknown="ignore"), [time_col]))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    rows: list[dict] = []
    for i in range(int(repeats)):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state + i)
        tr, te = next(splitter.split(data[feature_cols], y, groups=groups))

        train = data.iloc[tr]
        test = data.iloc[te]

        pipe.fit(train[feature_cols + ([time_col] if include_time_fixed_effects else [])], train["event_extinct_next_bin"])
        prob = pipe.predict_proba(test[feature_cols + ([time_col] if include_time_fixed_effects else [])])[:, 1]

        auc = float(roc_auc_score(test["event_extinct_next_bin"], prob))
        ap = float(average_precision_score(test["event_extinct_next_bin"], prob))

        coef = pipe.named_steps["clf"].coef_.reshape(-1)
        num_coef = coef[: len(feature_cols)]

        row = {"repeat": i, "auc": auc, "avg_precision": ap}
        for name, c in zip(feature_cols, num_coef, strict=True):
            row[f"coef_{name}"] = float(c)
            row[f"or_{name}"] = float(np.exp(c))
        rows.append(row)

    rep = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(data)),
        "event_rate": float(data["event_extinct_next_bin"].mean()),
        "auc_mean": float(rep["auc"].mean()),
        "auc_sd": float(rep["auc"].std(ddof=1)) if len(rep) > 1 else 0.0,
        "auc_p2_5": float(rep["auc"].quantile(0.025)),
        "auc_p97_5": float(rep["auc"].quantile(0.975)),
        "ap_mean": float(rep["avg_precision"].mean()),
        "ap_sd": float(rep["avg_precision"].std(ddof=1)) if len(rep) > 1 else 0.0,
    }

    coef_rows = []
    for name in feature_cols:
        coef_rows.append(
            {
                "feature": name,
                "coef_mean": float(rep[f"coef_{name}"].mean()),
                "coef_p2_5": float(rep[f"coef_{name}"].quantile(0.025)),
                "coef_p97_5": float(rep[f"coef_{name}"].quantile(0.975)),
                "odds_ratio_mean": float(rep[f"or_{name}"].mean()),
                "odds_ratio_p2_5": float(rep[f"or_{name}"].quantile(0.025)),
                "odds_ratio_p97_5": float(rep[f"or_{name}"].quantile(0.975)),
            }
        )
    coef_summary = pd.DataFrame(coef_rows).sort_values("odds_ratio_mean", ascending=False)

    return FitResult(repeats=rep, summary=summary, coef_summary=coef_summary)


def _filter_min_genus_bins(hazard: pd.DataFrame, genus_bin: pd.DataFrame, *, min_bins: int) -> pd.DataFrame:
    if min_bins <= 1:
        return hazard
    counts = genus_bin.groupby("genus")["time_bin"].nunique()
    keep = set(counts[counts >= int(min_bins)].index)
    return hazard[hazard["genus"].isin(keep)].copy()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", required=True, help="Scenario directory containing results/*.parquet")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--min-genus-bins", type=int, default=3, help="Filter to genera with at least this many bins")
    args = p.parse_args()

    scenario_dir = Path(args.scenario_dir)
    hazard_path = scenario_dir / "results" / "hazard_dataset.parquet"
    genus_bin_path = scenario_dir / "results" / "genus_bin.parquet"
    if not hazard_path.exists():
        raise FileNotFoundError(hazard_path)
    if not genus_bin_path.exists():
        raise FileNotFoundError(genus_bin_path)

    hazard = pd.read_parquet(hazard_path)
    genus_bin = pd.read_parquet(genus_bin_path, columns=["genus", "time_bin"])

    hazard = _filter_min_genus_bins(hazard, genus_bin, min_bins=int(args.min_genus_bins))
    hazard = _angle_component_features(hazard)

    # Common covariates.
    common = [
        "abundance",
        "geographic_range",
        "lat_range",
        "env_breadth",
        "age_bins",
    ]
    sampling_covars = ["global_velocity_km_per_myr_back", "n_occ", "n_localities"]

    standard_features = ["velocity_km_per_myr_back", "align_with_global_sampling", *sampling_covars, *common]
    decomposed_features = ["velocity_parallel_sampling", "velocity_orth_sampling", *sampling_covars, *common]

    # Drop any missing.
    needed = sorted(set(standard_features + decomposed_features + ["time_bin", "genus", "event_extinct_next_bin"]))
    hazard = hazard.dropna(subset=needed).copy()

    out_dir = scenario_dir / "results" / "posthoc"
    _ensure_dir(out_dir)

    runs = []
    for include_time in [True, False]:
        for model_name, feats in [("standard", standard_features), ("decomposed", decomposed_features)]:
            base_feats = [c for c in feats if c not in {"velocity_km_per_myr_back", "velocity_parallel_sampling", "velocity_orth_sampling", "align_with_global_sampling"}]

            fit_full = _fit_discrete_time_logit(
                hazard,
                feature_cols=feats,
                group_col="genus",
                time_col="time_bin",
                include_time_fixed_effects=include_time,
                repeats=args.repeats,
                random_state=args.seed,
            )
            fit_base = _fit_discrete_time_logit(
                hazard,
                feature_cols=base_feats,
                group_col="genus",
                time_col="time_bin",
                include_time_fixed_effects=include_time,
                repeats=args.repeats,
                random_state=args.seed,
            )

            tag = f"{model_name}_{'timeFE' if include_time else 'noTimeFE'}_minBins{args.min_genus_bins}"
            fit_full.repeats.to_csv(out_dir / f"{tag}_repeats.csv", index=False)
            fit_full.coef_summary.to_csv(out_dir / f"{tag}_coef_summary.csv", index=False)
            fit_base.coef_summary.to_csv(out_dir / f"{tag}_baseline_coef_summary.csv", index=False)

            meta = {
                "tag": tag,
                "n_rows": int(len(hazard)),
                "event_rate": float(hazard["event_extinct_next_bin"].mean()),
                "include_time_fixed_effects": bool(include_time),
                "model": model_name,
                "features": feats,
                "baseline_features": base_feats,
                "full": fit_full.summary,
                "baseline": fit_base.summary,
                "delta_auc": float(fit_full.summary["auc_mean"] - fit_base.summary["auc_mean"]),
            }
            _write_json(out_dir / f"{tag}_meta.json", meta)
            runs.append(meta)

            # Simple figure: OR for the mobility terms.
            coef = fit_full.coef_summary.set_index("feature")
            mob_feats = [c for c in feats if c.startswith("velocity") or c == "align_with_global_sampling"]
            mob_feats = [c for c in mob_feats if c in coef.index]
            if mob_feats:
                plot = coef.loc[mob_feats].copy()
                fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(plot) + 1.5))
                ax.errorbar(
                    plot["odds_ratio_mean"],
                    np.arange(len(plot)),
                    xerr=[
                        plot["odds_ratio_mean"] - plot["odds_ratio_p2_5"],
                        plot["odds_ratio_p97_5"] - plot["odds_ratio_mean"],
                    ],
                    fmt="o",
                    color="#1f77b4",
                    ecolor="gray",
                    capsize=3,
                )
                ax.axvline(1.0, color="black", linewidth=1, alpha=0.6)
                ax.set_yticks(np.arange(len(plot)))
                ax.set_yticklabels(plot.index.tolist())
                ax.set_xlabel("Odds ratio per 1 SD (mean ± 95% across splits)")
                ax.set_title(f"Mobility terms ({tag})")
                fig.tight_layout()
                fig.savefig(out_dir / f"{tag}_mobility_terms.png", dpi=200)
                plt.close(fig)

    (out_dir / "posthoc_summary.json").write_text(json.dumps(runs, indent=2) + "\n")
    pd.DataFrame(runs).to_csv(out_dir / "posthoc_summary.csv", index=False)
    print(f"Wrote: {out_dir / 'posthoc_summary.csv'}")


if __name__ == "__main__":
    main()

