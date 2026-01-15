from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _unit_xyz(lat_deg: float, lng_deg: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lng = np.deg2rad(lng_deg)
    return np.array(
        [np.cos(lat) * np.cos(lng), np.cos(lat) * np.sin(lng), np.sin(lat)],
        dtype=float,
    )


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    c = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return float(np.arccos(c))


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0088
    lat1 = np.deg2rad(lat1)
    lng1 = np.deg2rad(lng1)
    lat2 = np.deg2rad(lat2)
    lng2 = np.deg2rad(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def _build_discrete_time_hazard(genus_bin: pd.DataFrame) -> pd.DataFrame:
    time_bins = sorted(genus_bin["time_bin"].unique(), reverse=True)
    next_bin = {time_bins[i]: time_bins[i + 1] for i in range(len(time_bins) - 1)}

    df = genus_bin.copy()
    df["next_time_bin"] = df["time_bin"].map(next_bin)
    df = df.dropna(subset=["next_time_bin"]).copy()

    presence = genus_bin[["genus", "time_bin"]].drop_duplicates()
    presence = presence.rename(columns={"time_bin": "next_time_bin"}).assign(in_next=1)

    df = df.merge(presence, on=["genus", "next_time_bin"], how="left")
    df["event_extinct_next_bin"] = df["in_next"].isna().astype(int)
    return df.drop(columns=["in_next"])


def _fit_velocity_model(
    hazard: pd.DataFrame,
    *,
    feature_cols: list[str],
    repeats: int,
    seed: int,
) -> dict:
    y = hazard["event_extinct_next_bin"].astype(int)
    groups = hazard["genus"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
            ("time", OneHotEncoder(handle_unknown="ignore"), ["time_bin"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    rows = []
    for i in range(int(repeats)):
        split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed + i)
        tr, te = next(split.split(hazard[feature_cols], y, groups=groups))

        train = hazard.iloc[tr]
        test = hazard.iloc[te]

        cols = feature_cols + ["time_bin"]
        pipe.fit(train[cols], train["event_extinct_next_bin"])
        prob = pipe.predict_proba(test[cols])[:, 1]

        auc = float(roc_auc_score(test["event_extinct_next_bin"], prob))
        coef = pipe.named_steps["clf"].coef_.reshape(-1)[: len(feature_cols)]

        row = {"repeat": i, "auc": auc}
        for name, c in zip(feature_cols, coef, strict=True):
            row[f"coef_{name}"] = float(c)
            row[f"or_{name}"] = float(np.exp(c))
        rows.append(row)

    rep = pd.DataFrame(rows)
    summary = {
        "auc_mean": float(rep["auc"].mean()),
        "auc_sd": float(rep["auc"].std(ddof=1)) if len(rep) > 1 else 0.0,
        "velocity_or_mean": float(rep["or_velocity_km_per_myr_back"].mean()) if "or_velocity_km_per_myr_back" in rep else None,
        "velocity_or_p2_5": float(rep["or_velocity_km_per_myr_back"].quantile(0.025)) if "or_velocity_km_per_myr_back" in rep else None,
        "velocity_or_p97_5": float(rep["or_velocity_km_per_myr_back"].quantile(0.975)) if "or_velocity_km_per_myr_back" in rep else None,
    }
    return {"repeats": rep, "summary": summary}


def _permute_centroids_within_bins(genus_bin: pd.DataFrame, *, rng: np.random.Generator) -> pd.DataFrame:
    out = genus_bin.copy()
    for t, idx in out.groupby("time_bin").groups.items():
        idx_arr = np.array(list(idx), dtype=int)
        perm = rng.permutation(idx_arr)
        out.loc[idx_arr, ["centroid_lat", "centroid_lng"]] = out.loc[perm, ["centroid_lat", "centroid_lng"]].to_numpy()
    return out


def _recompute_velocity_and_alignment(
    genus_bin: pd.DataFrame,
    *,
    global_centroids: pd.DataFrame,
    max_delta_myr: float,
    velocity_cap_value: float,
) -> pd.DataFrame:
    gb = genus_bin.sort_values(["genus", "time_bin"], ascending=[True, False]).reset_index(drop=True).copy()
    gb["prev_time_bin"] = gb.groupby("genus")["time_bin"].shift(1)
    gb["prev_lat"] = gb.groupby("genus")["centroid_lat"].shift(1)
    gb["prev_lng"] = gb.groupby("genus")["centroid_lng"].shift(1)
    gb["delta_myr"] = gb["prev_time_bin"] - gb["time_bin"]

    valid = gb[["prev_lat", "prev_lng", "centroid_lat", "centroid_lng", "delta_myr"]].notna().all(axis=1)
    valid &= gb["delta_myr"] > 0
    valid &= gb["delta_myr"] <= float(max_delta_myr)

    dist = np.full(len(gb), np.nan, dtype=float)
    for i in valid.to_numpy().nonzero()[0]:
        dist[i] = _haversine_km(gb.loc[i, "prev_lat"], gb.loc[i, "prev_lng"], gb.loc[i, "centroid_lat"], gb.loc[i, "centroid_lng"])
    gb["distance_km_back"] = dist
    gb["velocity_km_per_myr_back"] = (gb["distance_km_back"] / gb["delta_myr"]).clip(upper=float(velocity_cap_value))

    g = global_centroids.set_index("time_bin")
    align = np.full(len(gb), np.nan, dtype=float)
    for i, row in gb.iterrows():
        if pd.isna(row["prev_lat"]) or pd.isna(row["prev_time_bin"]):
            continue
        t = float(row["time_bin"])
        if t not in g.index:
            continue
        g_row = g.loc[t]
        if pd.isna(g_row["prev_lat"]):
            continue
        u_prev = _unit_xyz(float(row["prev_lat"]), float(row["prev_lng"]))
        u_curr = _unit_xyz(float(row["centroid_lat"]), float(row["centroid_lng"]))
        d_tax = u_curr - u_prev

        g_prev = _unit_xyz(float(g_row["prev_lat"]), float(g_row["prev_lng"]))
        g_curr = _unit_xyz(float(g_row["global_centroid_lat"]), float(g_row["global_centroid_lng"]))
        d_glob = g_curr - g_prev

        angle = _angle_between(d_tax, d_glob)
        align[i] = float(np.cos(angle))

    gb["align_with_global_sampling"] = align
    return gb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", required=True)
    p.add_argument("--n-permutations", type=int, default=10)
    p.add_argument("--splits", type=int, default=10, help="Group splits per permutation")
    p.add_argument("--seed", type=int, default=202)
    args = p.parse_args()

    scenario_dir = Path(args.scenario_dir)
    genus_bin = pd.read_parquet(scenario_dir / "results" / "genus_bin.parquet")
    global_centroids = pd.read_csv(scenario_dir / "results" / "global_sampling_centroids.csv")

    meta = json.loads((scenario_dir / "results" / "meta.json").read_text())
    max_delta = float(meta["scenario"]["max_delta_myr"])
    velocity_cap = float(meta["velocity_cap_value"])

    # Observed fit (for comparison).
    hazard_obs = pd.read_parquet(scenario_dir / "results" / "hazard_dataset.parquet")
    features = [
        "velocity_km_per_myr_back",
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
    hazard_obs = hazard_obs.dropna(subset=features + ["time_bin", "genus", "event_extinct_next_bin"]).copy()
    observed = _fit_velocity_model(hazard_obs, feature_cols=features, repeats=int(args.splits), seed=int(args.seed))

    rng = np.random.default_rng(int(args.seed))
    perm_summaries = []
    perm_rows = []
    for p_i in range(int(args.n_permutations)):
        perm_rng = np.random.default_rng(int(args.seed) + 10_000 + p_i)
        gb_perm = _permute_centroids_within_bins(genus_bin, rng=perm_rng)
        gb_perm = _recompute_velocity_and_alignment(
            gb_perm,
            global_centroids=global_centroids,
            max_delta_myr=max_delta,
            velocity_cap_value=velocity_cap,
        )

        hazard = _build_discrete_time_hazard(gb_perm)
        hazard = hazard.dropna(subset=features + ["time_bin", "genus", "event_extinct_next_bin"]).copy()
        fit = _fit_velocity_model(hazard, feature_cols=features, repeats=int(args.splits), seed=int(args.seed) + 1000 * (p_i + 1))

        fit["repeats"]["permutation"] = p_i
        perm_rows.append(fit["repeats"])
        perm_summaries.append({"permutation": p_i, **fit["summary"]})

    out_dir = scenario_dir / "results" / "null_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perm_summaries).to_csv(out_dir / "centroid_perm_summary.csv", index=False)
    pd.concat(perm_rows, ignore_index=True).to_csv(out_dir / "centroid_perm_repeats.csv", index=False)
    _write = {
        "observed_summary": observed["summary"],
        "n_permutations": int(args.n_permutations),
        "splits_per_permutation": int(args.splits),
    }
    (out_dir / "centroid_perm_meta.json").write_text(json.dumps(_write, indent=2) + "\n")
    print(f"Wrote: {out_dir / 'centroid_perm_summary.csv'}")


if __name__ == "__main__":
    main()

