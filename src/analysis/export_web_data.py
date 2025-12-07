import pandas as pd
import numpy as np
import json
import os
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import analysis logic or re-implement simplified versions for export
# To ensure consistency, we'll re-implement the core logic here to output pure JSON structure.

def export_dashboard_data(data_path="data/processed/merged_occurrences.parquet", output_file="dashboard/web_data.json"):
    print(f"Exporting dashboard data from {data_path}...")
    
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Filter valid data
    df = df.dropna(subset=["mid_ma", "genus", "lat", "lng"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    
    # Create a subsample for heavy calculations (Modularity, ML)
    # 1.2M rows is too slow for real-time dashboard generation
    # Reduced to 20k for speed (approx 200 points per bin)
    if len(df) > 20000:
        print(f"Subsampling dataset from {len(df)} to 20,000 for heavy analysis...")
        df_heavy = df.sample(n=20000, random_state=42)
    else:
        df_heavy = df.copy()
    
    # --- 1. Diversity Curve ---
    diversity = df.groupby("time_bin")["genus"].nunique().sort_index(ascending=False)
    diversity_data = {
        "time": diversity.index.tolist(),
        "count": diversity.values.tolist()
    }
    
    # --- 2. Data Explorer (Unique Genera List) ---
    # User requested deduplicated list. Unique genera count is ~48k, which fits in JSON.
    print("Generating unique genera summary...")
    
    # Custom aggregation: get min/max/count age, plus a mode (most common) reference
    # Note: 'primary_reference' might contain NaN, so handle carefully
    def get_top_reference(series):
        try:
            return series.mode().iloc[0] if not series.mode().empty else "Unknown"
        except:
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
    
    sota_results = []
    df_heavy["lat_bin"] = (df_heavy["lat"] / 5).round() * 5
    df_heavy["lng_bin"] = (df_heavy["lng"] / 5).round() * 5
    df_heavy["locality"] = list(zip(df_heavy["lat_bin"], df_heavy["lng_bin"]))
    
    for time_bin, group in df_heavy.groupby("time_bin"):
        if len(group) < 50: continue
        
        # Modularity
        G = nx.Graph()
        localities = group["locality"].unique()
        genera = group["genus"].unique()
        if len(localities) < 5 or len(genera) < 5: continue
        
        G.add_nodes_from(localities, bipartite=0)
        G.add_nodes_from(genera, bipartite=1)
        G.add_edges_from(list(zip(group["locality"], group["genus"])))
        
        try:
            locality_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
            locality_graph = nx.bipartite.projected_graph(G, locality_nodes)
            communities = greedy_modularity_communities(locality_graph)
            modularity = nx.community.modularity(locality_graph, communities)
        except:
            modularity = None
            
        # Latitudinal Centroid
        mean_abs_lat = group["lat"].abs().mean()
        
        sota_results.append({
            "time": float(time_bin),
            "modularity": modularity,
            "mean_abs_lat": mean_abs_lat
        })
    
    sota_results.sort(key=lambda x: x["time"], reverse=True)
    
    # --- 3b. SQS Diversity ---
    # Simplified SQS calculation for export
    sqs_results = []
    quota = 0.5
    for time_bin, group in df.groupby("time_bin"):
        counts = group["genus"].value_counts()
        total_occ = counts.sum()
        if total_occ == 0: continue
        
        freqs = counts / total_occ
        freqs = freqs.sort_values(ascending=False)
        
        cum_freq = 0
        sqs_div = 0
        for f in freqs:
            cum_freq += f
            sqs_div += 1
            if cum_freq >= quota: break
        
        sqs_results.append({"time": float(time_bin), "sqs": sqs_div})
    
    sqs_results.sort(key=lambda x: x["time"], reverse=True)

    # --- 4. ML Extinction ---
    # Re-run simplified ML
    time_bins = sorted(df_heavy["time_bin"].unique(), reverse=True)
    ml_records = []
    
    for i, current_bin in enumerate(time_bins[:-1]):
        next_bin = time_bins[i + 1]
        current_data = df_heavy[df_heavy["time_bin"] == current_bin]
        next_data = df_heavy[df_heavy["time_bin"] == next_bin]
        next_genera = set(next_data["genus"].unique())
        
        for genus in current_data["genus"].unique():
            genus_data = current_data[current_data["genus"] == genus]
            older_bins = [b for b in time_bins if b > current_bin]
            age = sum(1 for b in older_bins if genus in df_heavy[df_heavy["time_bin"] == b]["genus"].values)
            
            ml_records.append({
                "geographic_range": genus_data["locality"].nunique(),
                "abundance": len(genus_data),
                "lat_range": genus_data["lat"].max() - genus_data["lat"].min(),
                "age": age,
                "extinct": 1 if genus not in next_genera else 0
            })
            
    ml_df = pd.DataFrame(ml_records).fillna(0)
    ml_data = {}
    
    if len(ml_df) > 100:
        X = ml_df[["geographic_range", "abundance", "lat_range", "age"]]
        y = ml_df["extinct"]
        clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        clf.fit(X, y)
        ml_data = {
            "features": ["Geographic Range", "Abundance", "Latitudinal Range", "Age"],
            "importance": clf.feature_importances_.tolist(),
            "accuracy": clf.score(X, y) # Just training score for dashboard display
        }

    # --- Final JSON ---
    final_data = {
        "diversity": diversity_data,
        "sqs": {
            "time": [r["time"] for r in sqs_results],
            "count": [r["sqs"] for r in sqs_results]
        },
        "map": explorer_data,
        "sota": {
            "time": [r["time"] for r in sota_results],
            "modularity": [r["modularity"] for r in sota_results],
            "mean_abs_lat": [r["mean_abs_lat"] for r in sota_results]
        },
        "ml": ml_data
    }
    
    with open(output_file, "w") as f:
        json.dump(final_data, f)
    
    print(f"Dashboard data saved to {output_file}")

if __name__ == "__main__":
    export_dashboard_data()
