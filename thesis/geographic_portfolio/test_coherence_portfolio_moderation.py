from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis.paleobiotic_velocity.run_pipeline import compute_genus_bin_table, load_occurrences  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _classify_env(env: Any) -> str:
    if env is None or (isinstance(env, float) and not np.isfinite(env)):
        return "unknown"
    s = str(env).strip().lower()
    if not s:
        return "unknown"
    terrestrial_terms = [
        "terrestrial",
        "fluvial",
        "lacustrine",
        "delta",
        "freshwater",
        "non-marine",
        "nonmarine",
        "eolian",
        "loess",
        "soil",
        "cave",
        "spring",
        "swamp",
        "paludal",
        "floodplain",
        "karst",
    ]
    if any(t in s for t in terrestrial_terms):
        return "terrestrial"
    marine_terms = [
        "marine",
        "reef",
        "subtidal",
        "offshore",
        "shelf",
        "basinal",
        "slope",
        "lagoon",
        "open",
        "deep",
        "carbonate",
        "platform",
        "pelagic",
        "ocean",
        "intertidal",
        "coastal",
    ]
    if any(t in s for t in marine_terms):
        return "marine"
    return "other"


def fit_repeated_group_logit(
    data: pd.DataFrame,
    *,
    target: str,
    group: str,
    numeric_features: list[str],
    categorical_features: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    cols = numeric_features + categorical_features + [target, group]
    df = data.dropna(subset=cols).copy()
    y = df[target].astype(int).to_numpy()
    groups = df[group].astype(float).to_numpy()

    transformers = [("num", StandardScaler(), numeric_features)]
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    splitter = GroupShuffleSplit(n_splits=int(repeats), test_size=0.3, random_state=int(seed))
    rows = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups=groups)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        pipe.fit(train[numeric_features + categorical_features], train[target])
        prob = pipe.predict_proba(test[numeric_features + categorical_features])[:, 1]
        auc = float(roc_auc_score(test[target], prob)) if len(np.unique(test[target])) > 1 else float("nan")
        ap = float(average_precision_score(test[target], prob))

        coef = pipe.named_steps["clf"].coef_.reshape(-1)
        num_coef = coef[: len(numeric_features)]
        row = {"repeat": i, "auc": auc, "avg_precision": ap, "n_train": int(len(train)), "n_test": int(len(test))}
        for name, c in zip(numeric_features, num_coef, strict=True):
            row[f"coef_{name}"] = float(c)
            row[f"or_{name}"] = float(np.exp(c))
        rows.append(row)

    rep = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(df)),
        "event_rate": float(df[target].mean()),
        "auc_mean": float(rep["auc"].mean()),
        "auc_sd": float(rep["auc"].std(ddof=1)) if len(rep) > 1 else 0.0,
        "ap_mean": float(rep["avg_precision"].mean()),
        "ap_sd": float(rep["avg_precision"].std(ddof=1)) if len(rep) > 1 else 0.0,
    }

    coef_rows = []
    for name in numeric_features:
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
    return {"repeats": rep, "summary": summary, "coef_summary": coef_summary}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/merged_occurrences.parquet")
    p.add_argument("--earth", default="thesis/earth_system/climate_540myr/output/climate_540myr_timeseries.csv")
    p.add_argument("--out", default="thesis/geographic_portfolio/output_coherence_moderation")
    p.add_argument("--coords-mode", choices=["paleo", "modern"], default="paleo")
    p.add_argument("--realm", choices=["marine", "terrestrial", "all"], default="marine")
    p.add_argument("--time-bin-myr", type=float, default=10.0)
    p.add_argument("--grid-deg", type=float, default=5.0)
    p.add_argument("--min_abundance", type=int, default=3)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--with-phylum", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "results")

    print("Loading occurrences…")
    occ = load_occurrences(
        args.data,
        coords_mode=args.coords_mode,  # type: ignore[arg-type]
        time_bin_myr=float(args.time_bin_myr),
        locality_bin_deg=float(args.grid_deg),
    )
    occ = occ[occ["source_db"] == "PBDB"].copy()
    occ["env_class"] = occ["environment"].map(_classify_env)
    if args.realm != "all":
        occ = occ[occ["env_class"] == args.realm].copy()

    print("Building genus×bin table…")
    gb = compute_genus_bin_table(occ, weighting="locality")
    gb["log_abundance"] = np.log1p(gb["abundance"].astype(float))
    gb["log_geographic_range"] = np.log1p(gb["geographic_range"].astype(float))
    gb["log_env_breadth"] = np.log1p(gb["env_breadth"].astype(float))
    gb["centroid_abs_lat"] = gb["centroid_lat"].abs()

    gb = gb[gb["abundance"] >= int(args.min_abundance)].copy()

    # Define survival to the next (younger) bin.
    step = float(args.time_bin_myr)
    gb["post_bin"] = gb["time_bin"] - step
    next_keys = set(zip(gb["genus"].astype(str), gb["time_bin"].astype(float), strict=False))
    gb["survived_next"] = [
        int((str(gen), float(post)) in next_keys) for gen, post in zip(gb["genus"], gb["post_bin"], strict=True)
    ]

    # Join climate forcing for the transition into post_bin (delta_from_prev is defined at the younger time).
    earth = pd.read_csv(args.earth).rename(columns={"time_ma": "time_bin"}).copy()
    earth = earth.add_prefix("earth_")
    earth = earth.rename(columns={"earth_time_bin": "post_bin"})
    gb = gb.merge(earth, on="post_bin", how="left")

    # Interaction term: does range protect less when forcing is more coherent?
    # Coherence metric is in [0,1] (sign agreement); higher means more spatially synchronized sign.
    gb["range_x_coh_sign"] = gb["log_geographic_range"] * gb["earth_delta_from_prev_T_sign_agreement_frac"]

    dataset_path = out_dir / "dataset.parquet"
    gb.to_parquet(dataset_path, index=False)

    numeric = [
        "log_abundance",
        "log_geographic_range",
        "lat_range",
        "log_env_breadth",
        "centroid_abs_lat",
        "time_bin",
        "earth_delta_from_prev_T_field_meanabs",
        "earth_delta_from_prev_T_sign_agreement_frac",
        "range_x_coh_sign",
    ]
    categorical = ["phylum"] if args.with_phylum else []

    fit = fit_repeated_group_logit(
        gb,
        target="survived_next",
        group="post_bin",
        numeric_features=numeric,
        categorical_features=categorical,
        repeats=int(args.repeats),
        seed=int(args.seed),
    )

    fit["repeats"].to_csv(out_dir / "results" / "repeats.csv", index=False)
    fit["coef_summary"].to_csv(out_dir / "results" / "coef_summary.csv", index=False)
    meta = {
        "coords_mode": args.coords_mode,
        "realm": args.realm,
        "time_bin_myr": float(args.time_bin_myr),
        "grid_deg": float(args.grid_deg),
        "min_abundance": int(args.min_abundance),
        "with_phylum": bool(args.with_phylum),
        "summary": fit["summary"],
    }
    (out_dir / "results" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Markdown summary.
    cs = fit["coef_summary"].set_index("feature")
    def _row(name: str) -> str:
        if name not in cs.index:
            return f"- {name}: (missing)"
        r = cs.loc[name].to_dict()
        return (
            f"- {name}: OR={r['odds_ratio_mean']:.3f} (95% {r['odds_ratio_p2_5']:.3f}–{r['odds_ratio_p97_5']:.3f})"
        )

    lines = [
        "# Coherence moderation test (portfolio selectivity; exploratory)",
        "",
        "Outcome: genus survives to the next 10 Myr bin (`survived_next`).",
        "Key hypothesis: the survival benefit of geographic range weakens when forcing is spatially coherent.",
        "",
        f"- dataset: `{dataset_path}`",
        f"- rows: {fit['summary']['n_rows']:,}",
        f"- event rate: {fit['summary']['event_rate']:.3f}",
        f"- grouped CV: GroupShuffleSplit by `post_bin` (held-out transitions), repeats={int(args.repeats)}",
        f"- AUC mean±sd: {fit['summary']['auc_mean']:.3f}±{fit['summary']['auc_sd']:.3f}",
        "",
        "## Coefficients (odds ratios per 1 SD; mean ± 95% across splits)",
        "",
        _row("log_geographic_range"),
        _row("earth_delta_from_prev_T_sign_agreement_frac"),
        _row("earth_delta_from_prev_T_field_meanabs"),
        _row("range_x_coh_sign"),
        "",
        "Interpretation note:",
        "- If `range_x_coh_sign` has OR < 1, range becomes *less protective* as coherence increases (supports the moderation claim).",
        "",
        "## Outputs",
        "",
        f"- coef summary: `{out_dir / 'results' / 'coef_summary.csv'}`",
        f"- repeats: `{out_dir / 'results' / 'repeats.csv'}`",
        f"- meta: `{out_dir / 'results' / 'meta.json'}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

