import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from src.analysis.geo import add_analysis_coordinates, add_binned_locality
from src.analysis.provinciality import compute_locality_network_modularity
from src.analysis.cleaning import clean_taxon_series

# Import analysis logic or re-implement simplified versions for export
# To ensure consistency, we'll re-implement the core logic here to output pure JSON structure.

def export_dashboard_data(
    data_path="data/processed/merged_occurrences.parquet",
    output_file="dashboard/web_data.json",
    *,
    explorer_output_file: str | None = "dashboard/explorer_data.json",
):
    print(f"Exporting dashboard data from {data_path}...")
    
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Filter valid data (coordinates handled via analysis_lat/lng below)
    df["genus"] = clean_taxon_series(df["genus"])
    df = df.dropna(subset=["mid_ma", "genus"])
    df["mid_ma"] = pd.to_numeric(df["mid_ma"], errors="coerce")
    df = df.dropna(subset=["mid_ma"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5

    # Prefer paleocoordinates when present; fall back to modern coords.
    df = add_analysis_coordinates(df)
    df_geo = df.dropna(subset=["analysis_lat", "analysis_lng"]).copy()
    
    # --- 1. Diversity Curve ---
    diversity = df.groupby("time_bin")["genus"].nunique().sort_index(ascending=False)
    diversity_data = {
        "time": diversity.index.tolist(),
        "count": diversity.values.tolist()
    }
    
    # --- 2. Data Explorer (Unique Genera List) ---
    # User requested deduplicated list. Unique genera count is ~48k, which fits in JSON.
    print("Generating unique genera summary...")
    if "primary_reference" not in df.columns:
        df["primary_reference"] = "Unknown"
    
    # Custom aggregation: get min/max/count age, plus a mode (most common) reference
    # Note: 'primary_reference' might contain NaN, so handle carefully
    def get_top_reference(series):
        try:
            return series.mode().iloc[0] if not series.mode().empty else "Unknown"
        except Exception:
            return "Unknown"

    genus_summary = df.groupby("genus").agg({
        "mid_ma": ["min", "max", "count"],
        "primary_reference": lambda x: get_top_reference(x)
    }).reset_index()
    
    # Flatten columns
    genus_summary.columns = ["genus", "min_age", "max_age", "count", "reference"]
    
    # Sort by count (finding the most common/famous ones first usually)
    genus_summary = genus_summary.sort_values("count", ascending=False)
    
    explorer_data = {
        "genus": genus_summary["genus"].tolist(),
        "min_age": genus_summary["min_age"].tolist(),
        "max_age": genus_summary["max_age"].tolist(),
        "count": genus_summary["count"].tolist(),
        "reference": genus_summary["reference"].tolist()
    }
    
    # --- 3. SOTA: Provinciality over time (network modularity) ---
    # Goal: reduce jitter from uneven sampling by using a fixed per-bin sample size.
    TIME_BIN_WIDTH_MA = 5.0
    LOCALITY_BIN_DEG = 5.0
    MIN_OCC_PER_BIN = 200
    MAX_OCC_SAMPLE_PER_BIN = 5000
    SMOOTH_WINDOW_MYR = 50.0
    smooth_window_bins = max(1, int(round(SMOOTH_WINDOW_MYR / TIME_BIN_WIDTH_MA)))

    sota_results = []

    for time_bin, group in df_geo.groupby("time_bin"):
        n_occ_total = int(len(group))
        if n_occ_total < MIN_OCC_PER_BIN:
            continue

        # Fixed-size sampling per bin for comparability and stability.
        n_sample = min(MAX_OCC_SAMPLE_PER_BIN, n_occ_total)
        group_sample = group.sample(n=n_sample, random_state=42 + int(time_bin))
        group_sample = add_binned_locality(
            group_sample,
            lat_col="analysis_lat",
            lng_col="analysis_lng",
            bin_degrees=LOCALITY_BIN_DEG,
            locality_col="locality",
        )

        mod_res = compute_locality_network_modularity(group_sample, locality_col="locality", genus_col="genus")
        mean_abs_lat = float(group_sample["analysis_lat"].abs().mean())

        sota_results.append(
            {
                "time": float(time_bin),
                "modularity": mod_res.modularity,
                "mean_abs_lat": mean_abs_lat,
                "n_occ_total": n_occ_total,
                "n_occ_sample": int(n_sample),
                "n_unique_edges": mod_res.n_unique_edges,
                "n_localities": mod_res.n_localities,
                "n_genera": mod_res.n_genera,
            }
        )

    if sota_results:
        sota_df = pd.DataFrame(sota_results).sort_values("time", ascending=False)
        sota_df["modularity_smooth"] = (
            sota_df["modularity"].astype(float).rolling(smooth_window_bins, center=True, min_periods=1).mean()
        )
        sota_df["mean_abs_lat_smooth"] = (
            sota_df["mean_abs_lat"].astype(float).rolling(smooth_window_bins, center=True, min_periods=1).mean()
        )
    else:
        sota_df = pd.DataFrame(
            columns=[
                "time",
                "modularity",
                "modularity_smooth",
                "mean_abs_lat",
                "mean_abs_lat_smooth",
                "n_occ_total",
                "n_occ_sample",
                "n_localities",
                "n_genera",
            ]
        )

    # --- 3b. SQS Diversity ---
    # Simplified SQS calculation for export
    sqs_results = []
    quota = 0.5
    for time_bin, group in df.groupby("time_bin"):
        counts = group["genus"].value_counts()
        total_occ = counts.sum()
        if total_occ == 0:
            continue
        
        freqs = counts / total_occ
        freqs = freqs.sort_values(ascending=False)
        
        cum_freq = 0
        sqs_div = 0
        for f in freqs:
            cum_freq += f
            sqs_div += 1
            if cum_freq >= quota:
                break
        
        sqs_results.append({"time": float(time_bin), "sqs": sqs_div})
    
    sqs_results.sort(key=lambda x: x["time"], reverse=True)

    # --- 4. ML Extinction ---
    # Build a lightweight training set: sample occurrences (not rows) to keep dashboard generation fast.
    MAX_ML_ROWS = 50_000
    df_ml = df_geo
    if len(df_ml) > MAX_ML_ROWS:
        print(f"Subsampling dataset from {len(df_ml)} to {MAX_ML_ROWS:,} for ML...")
        df_ml = df_ml.sample(n=MAX_ML_ROWS, random_state=42).copy()
    else:
        df_ml = df_ml.copy()

    df_ml = add_binned_locality(df_ml, bin_degrees=LOCALITY_BIN_DEG, locality_col="locality")

    # Per-genus-per-bin features
    agg = (
        df_ml.groupby(["time_bin", "genus"])
        .agg(
            geographic_range=("locality", "nunique"),
            abundance=("genus", "size"),
            lat_min=("analysis_lat", "min"),
            lat_max=("analysis_lat", "max"),
        )
        .reset_index()
    )
    agg["lat_range"] = agg["lat_max"] - agg["lat_min"]

    # "Age" = number of older bins the genus has already appeared in (within this sampled dataset)
    presence = df_ml[["genus", "time_bin"]].drop_duplicates()
    presence = presence.sort_values(["genus", "time_bin"], ascending=[True, False])
    presence["age"] = presence.groupby("genus").cumcount()
    agg = agg.merge(presence, on=["genus", "time_bin"], how="left")

    # Target: extinct in next time bin?
    time_bins = sorted(df_ml["time_bin"].unique(), reverse=True)
    next_bin_map = {time_bins[i]: time_bins[i + 1] for i in range(len(time_bins) - 1)}
    agg["next_bin"] = agg["time_bin"].map(next_bin_map)
    agg = agg.dropna(subset=["next_bin"])
    next_presence = presence.rename(columns={"time_bin": "next_bin"}).assign(in_next=1)[["genus", "next_bin", "in_next"]]
    agg = agg.merge(next_presence, on=["genus", "next_bin"], how="left")
    agg["extinct"] = (agg["in_next"].isna()).astype(int)

    feature_cols = ["geographic_range", "abundance", "lat_range", "age"]
    ml_data: dict = {}
    if len(agg) > 200:
        X = agg[feature_cols].fillna(0)
        y = agg["extinct"]
        groups = agg["genus"]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            roc_auc = 0.5

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        else:
            tn = fp = fn = tp = 0

        ml_data = {
            "features": ["Geographic Range", "Abundance", "Latitudinal Range", "Age (bins)"],
            "importance": clf.feature_importances_.tolist(),
            "metrics": {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "roc_auc": roc_auc,
                "n_samples": int(len(agg)),
                "extinction_rate": float(y.mean()),
                "holdout_fraction": 0.3,
                "holdout_split": "grouped_by_genus",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_genera_train": int(agg.loc[train_idx, "genus"].nunique()),
                "n_genera_test": int(agg.loc[test_idx, "genus"].nunique()),
                "confusion_matrix": {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                },
            },
        }

    # --- Final JSON ---
    summary_data = {
        "diversity": diversity_data,
        "sqs": {
            "time": [r["time"] for r in sqs_results],
            "count": [r["sqs"] for r in sqs_results]
        },
        "sota": {
            "time": sota_df["time"].astype(float).tolist() if len(sota_df) else [],
            "modularity": sota_df["modularity"].tolist() if len(sota_df) else [],
            "modularity_smooth": sota_df["modularity_smooth"].tolist() if len(sota_df) else [],
            "mean_abs_lat": sota_df["mean_abs_lat"].astype(float).tolist() if len(sota_df) else [],
            "mean_abs_lat_smooth": sota_df["mean_abs_lat_smooth"].astype(float).tolist() if len(sota_df) else [],
            "n_occ_total": sota_df["n_occ_total"].astype(int).tolist() if len(sota_df) else [],
            "n_occ_sample": sota_df["n_occ_sample"].astype(int).tolist() if len(sota_df) else [],
            "n_localities": sota_df["n_localities"].astype(int).tolist() if len(sota_df) else [],
            "n_genera": sota_df["n_genera"].astype(int).tolist() if len(sota_df) else [],
            "params": {
                "time_bin_width_ma": TIME_BIN_WIDTH_MA,
                "locality_bin_deg": LOCALITY_BIN_DEG,
                "min_occ_per_bin": MIN_OCC_PER_BIN,
                "max_occ_sample_per_bin": MAX_OCC_SAMPLE_PER_BIN,
                "smooth_window_myr": SMOOTH_WINDOW_MYR,
            },
        },
        "ml": ml_data
    }
    
    with open(output_file, "w") as f:
        json.dump(summary_data, f)

    if explorer_output_file is not None:
        with open(explorer_output_file, "w") as f:
            json.dump(explorer_data, f)
    
    print(f"Dashboard data saved to {output_file}")
    if explorer_output_file is not None:
        print(f"Explorer data saved to {explorer_output_file}")

if __name__ == "__main__":
    export_dashboard_data()
