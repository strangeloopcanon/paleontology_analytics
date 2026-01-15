from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _filter_min_genus_bins(hazard: pd.DataFrame, genus_bin: pd.DataFrame, *, min_bins: int) -> pd.DataFrame:
    if min_bins <= 1:
        return hazard
    counts = genus_bin.groupby("genus")["time_bin"].nunique()
    keep = set(counts[counts >= int(min_bins)].index)
    return hazard[hazard["genus"].isin(keep)].copy()


def _add_crisis_indicators(df: pd.DataFrame, *, window_myr: float = 10.0) -> pd.DataFrame:
    """
    Coarse crisis windows (Ma) for exploratory interaction tests.

    Ages approximate the major 'Big Five' boundaries within the PBDB interval used here.
    """
    crises = {
        "end_ordovician": 444.0,
        "late_devonian": 372.0,
        "end_permian": 252.0,
        "end_triassic": 201.0,
        # end_cretaceous ~66 Ma is at the edge of the default dataset.
    }
    out = df.copy()
    for name, age in crises.items():
        out[f"crisis_{name}"] = (out["time_bin"].astype(float) - float(age)).abs() <= float(window_myr)
        out[f"crisis_{name}"] = out[f"crisis_{name}"].astype(int)
    crisis_cols = [f"crisis_{k}" for k in crises]
    out["crisis_any"] = out[crisis_cols].max(axis=1)
    return out


@dataclass(frozen=True)
class Fit:
    repeats: pd.DataFrame
    summary: dict
    coef_summary: pd.DataFrame


def _fit(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    include_time_fixed_effects: bool,
    repeats: int,
    random_state: int,
) -> Fit:
    y = data["event_extinct_next_bin"].astype(int)
    groups = data["genus"]

    transformers: list[tuple[str, object, list[str]]] = [("num", StandardScaler(), feature_cols)]
    if include_time_fixed_effects:
        transformers.append(("time", OneHotEncoder(handle_unknown="ignore"), ["time_bin"]))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    rows = []
    for i in range(int(repeats)):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state + i)
        tr, te = next(splitter.split(data[feature_cols], y, groups=groups))

        train = data.iloc[tr]
        test = data.iloc[te]

        cols = feature_cols + (["time_bin"] if include_time_fixed_effects else [])
        pipe.fit(train[cols], train["event_extinct_next_bin"])
        prob = pipe.predict_proba(test[cols])[:, 1]

        auc = float(roc_auc_score(test["event_extinct_next_bin"], prob))
        ap = float(average_precision_score(test["event_extinct_next_bin"], prob))

        coef = pipe.named_steps["clf"].coef_.reshape(-1)[: len(feature_cols)]
        row = {"repeat": i, "auc": auc, "avg_precision": ap}
        for name, c in zip(feature_cols, coef, strict=True):
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

    return Fit(repeats=rep, summary=summary, coef_summary=coef_summary)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", required=True)
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--min-genus-bins", type=int, default=3)
    p.add_argument("--window-myr", type=float, default=10.0)
    p.add_argument("--time-fe", action="store_true", help="Include time-bin fixed effects")
    args = p.parse_args()

    scenario_dir = Path(args.scenario_dir)
    hazard = pd.read_parquet(scenario_dir / "results" / "hazard_dataset.parquet")
    genus_bin = pd.read_parquet(scenario_dir / "results" / "genus_bin.parquet", columns=["genus", "time_bin"])

    hazard = _filter_min_genus_bins(hazard, genus_bin, min_bins=int(args.min_genus_bins))
    hazard = _add_crisis_indicators(hazard, window_myr=float(args.window_myr))

    # Interaction term(s).
    hazard["vel_x_crisis_any"] = hazard["velocity_km_per_myr_back"].astype(float) * hazard["crisis_any"].astype(float)

    base_covars = [
        "abundance",
        "geographic_range",
        "lat_range",
        "env_breadth",
        "age_bins",
        "global_velocity_km_per_myr_back",
        "n_occ",
        "n_localities",
        "align_with_global_sampling",
    ]

    model_cols = ["velocity_km_per_myr_back", "crisis_any", "vel_x_crisis_any", *base_covars]
    baseline_cols = ["crisis_any", *base_covars]

    need = sorted(set(model_cols + baseline_cols + ["time_bin", "genus", "event_extinct_next_bin"]))
    hazard = hazard.dropna(subset=need).copy()

    out_dir = scenario_dir / "results" / "event_interactions"
    _ensure_dir(out_dir)

    fit_full = _fit(
        hazard,
        feature_cols=model_cols,
        include_time_fixed_effects=bool(args.time_fe),
        repeats=args.repeats,
        random_state=args.seed,
    )
    fit_base = _fit(
        hazard,
        feature_cols=baseline_cols,
        include_time_fixed_effects=bool(args.time_fe),
        repeats=args.repeats,
        random_state=args.seed,
    )

    tag = f"crisisAny_window{args.window_myr:g}myr_{'timeFE' if args.time_fe else 'noTimeFE'}_minBins{args.min_genus_bins}"
    fit_full.repeats.to_csv(out_dir / f"{tag}_repeats.csv", index=False)
    fit_full.coef_summary.to_csv(out_dir / f"{tag}_coef_summary.csv", index=False)
    fit_base.coef_summary.to_csv(out_dir / f"{tag}_baseline_coef_summary.csv", index=False)

    meta = {
        "tag": tag,
        "n_rows": int(len(hazard)),
        "event_rate": float(hazard["event_extinct_next_bin"].mean()),
        "time_fe": bool(args.time_fe),
        "min_genus_bins": int(args.min_genus_bins),
        "window_myr": float(args.window_myr),
        "full": fit_full.summary,
        "baseline": fit_base.summary,
        "delta_auc": float(fit_full.summary["auc_mean"] - fit_base.summary["auc_mean"]),
    }
    (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote: {out_dir / f'{tag}_meta.json'}")


if __name__ == "__main__":
    main()

