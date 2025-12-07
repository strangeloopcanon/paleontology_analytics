import pandas as pd
import numpy as np
import json
import os

def calculate_rates(data_path="data/processed/merged_occurrences.parquet", output_file="dashboard/rates_data.json"):
    """
    Calculate origination and extinction rates per time bin.
    
    - Origination Rate = (New genera in bin) / (Total genera in bin)
    - Extinction Rate = (Genera that disappear after bin) / (Total genera in bin)
    """
    print("Calculating origination/extinction rates...")
    
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["mid_ma", "genus"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    
    time_bins = sorted(df["time_bin"].unique(), reverse=True)  # Oldest first
    
    results = []
    prev_genera = set()
    
    for i, current_bin in enumerate(time_bins):
        current_genera = set(df[df["time_bin"] == current_bin]["genus"].unique())
        
        if i < len(time_bins) - 1:
            next_bin = time_bins[i + 1]
            next_genera = set(df[df["time_bin"] == next_bin]["genus"].unique())
        else:
            next_genera = set()
        
        total = len(current_genera)
        if total == 0:
            continue
            
        # Originations: genera in current bin that weren't in previous (older) bin
        originations = len(current_genera - prev_genera)
        
        # Extinctions: genera in current bin that aren't in next (younger) bin
        extinctions = len(current_genera - next_genera)
        
        orig_rate = originations / total
        ext_rate = extinctions / total
        
        results.append({
            "time": float(current_bin),
            "origination_rate": orig_rate,
            "extinction_rate": ext_rate,
            "total_genera": total,
            "originations": originations,
            "extinctions": extinctions
        })
        
        prev_genera = current_genera
    
    # Detect mass extinctions (extinction rate > 2 std above mean)
    ext_rates = [r["extinction_rate"] for r in results if r["extinction_rate"] is not None]
    mean_ext = np.mean(ext_rates)
    std_ext = np.std(ext_rates)
    threshold = mean_ext + 2 * std_ext
    
    for r in results:
        r["is_mass_extinction"] = bool(r["extinction_rate"] > threshold)
    
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    print(f"Rates data saved to {output_file}")
    return results


def calculate_climate_correlation(data_path="data/processed/merged_occurrences.parquet", output_file="dashboard/climate_data.json"):
    """
    Correlate diversity with Phanerozoic temperature proxy (δ18O).
    Using simplified Veizer curve approximation.
    """
    print("Calculating climate correlation...")
    
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["mid_ma", "genus"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    
    # Diversity per bin
    diversity = df.groupby("time_bin")["genus"].nunique()
    
    # Improved Phanerozoic Temperature Curve (Approximate Global Avg Temp in °C)
    # Based on Scotese (2021) / Veizer (2000)
    # Time (Ma) -> Temp (°C)
    temp_points = {
        0: 14, 10: 14, 30: 18, 50: 24, 65: 22,  # Cenozoic
        80: 24, 100: 26, 140: 22,               # Cretaceous
        170: 20, 200: 19,                       # Jurassic
        230: 22, 250: 25,                       # Triassic (Hot!)
        270: 16, 300: 12,                       # Permian/Carboniferous (Cold)
        340: 14, 360: 20,                       # Devonian
        400: 22, 420: 20,                       # Silurian
        440: 16, 450: 12,                       # Ordovician (Glaciation)
        480: 18, 500: 22, 540: 24               # Cambrian
    }
    
    sorted_times = sorted(temp_points.keys())
    
    def get_temp(age):
        # Linear interpolation
        if age <= sorted_times[0]: return temp_points[sorted_times[0]]
        if age >= sorted_times[-1]: return temp_points[sorted_times[-1]]
        
        for i in range(len(sorted_times) - 1):
            t1, t2 = sorted_times[i], sorted_times[i+1]
            if t1 <= age <= t2:
                temp1, temp2 = temp_points[t1], temp_points[t2]
                fraction = (age - t1) / (t2 - t1)
                return temp1 + (temp2 - temp1) * fraction
        return 14

    # Generate high-resolution temperature curve for plotting (every 1 Ma)
    high_res_temp = []
    for t in range(0, 541):
        high_res_temp.append({
            "time": t,
            "temperature": get_temp(t)
        })

    results = []
    for time_bin in diversity.index:
        results.append({
            "time": float(time_bin),
            "diversity": int(diversity[time_bin]),
            "temperature": get_temp(time_bin) # Keep for correlation calc
        })
    
    results.sort(key=lambda x: x["time"], reverse=True)
    
    # Calculate correlation
    df_corr = pd.DataFrame(results)
    correlation = df_corr["diversity"].corr(df_corr["temperature"])
    
    with open(output_file, "w") as f:
        json.dump({
            "timeseries": results,
            "temperature_curve": high_res_temp, # New high-res data
            "correlation": correlation if not np.isnan(correlation) else 0.0
        }, f)
    
    print(f"Climate data saved to {output_file}. Correlation: {correlation:.3f}")
    return {"timeseries": results, "temperature_curve": high_res_temp, "correlation": correlation if not np.isnan(correlation) else 0.0}


def calculate_null_model(data_path="data/processed/merged_occurrences.parquet", output_file="dashboard/null_model_data.json", n_iterations=100):
    """
    Generate null distribution for modularity to test significance.
    """
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    
    print(f"Running null model test ({n_iterations} iterations)...")
    
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["mid_ma", "genus", "lat", "lng"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    df["lat_bin"] = (df["lat"] / 5).round() * 5
    df["lng_bin"] = (df["lng"] / 5).round() * 5
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))
    
    # Pick a representative time bin with good data
    bin_sizes = df.groupby("time_bin").size()
    target_bin = bin_sizes.idxmax()  # Use the bin with most data
    
    group = df[df["time_bin"] == target_bin]
    localities = list(group["locality"].unique())
    genera = list(group["genus"].unique())
    
    if len(localities) < 10 or len(genera) < 10:
        print("Insufficient data for null model test")
        return
    
    # Build real network
    G = nx.Graph()
    G.add_nodes_from(localities, bipartite=0)
    G.add_nodes_from(genera, bipartite=1)
    G.add_edges_from(list(zip(group["locality"], group["genus"])))
    
    locality_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    locality_graph = nx.bipartite.projected_graph(G, locality_nodes)
    
    communities = greedy_modularity_communities(locality_graph)
    observed_modularity = nx.community.modularity(locality_graph, communities)
    
    # Null distribution: shuffle genus assignments
    null_modularities = []
    edges = list(zip(group["locality"], group["genus"]))
    
    for _ in range(n_iterations):
        shuffled_genera = np.random.permutation(group["genus"].values)
        shuffled_edges = list(zip(group["locality"], shuffled_genera))
        
        G_null = nx.Graph()
        G_null.add_nodes_from(localities, bipartite=0)
        G_null.add_nodes_from(genera, bipartite=1)
        G_null.add_edges_from(shuffled_edges)
        
        try:
            null_locality_graph = nx.bipartite.projected_graph(G_null, locality_nodes)
            null_communities = greedy_modularity_communities(null_locality_graph)
            null_mod = nx.community.modularity(null_locality_graph, null_communities)
            null_modularities.append(null_mod)
        except:
            pass
    
    # Calculate p-value
    p_value = sum(1 for m in null_modularities if m >= observed_modularity) / len(null_modularities)
    
    output = {
        "observed_modularity": observed_modularity,
        "null_distribution": null_modularities,
        "p_value": p_value,
        "time_bin": float(target_bin),
        "significant": p_value < 0.05
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f)
    
    print(f"Null model saved. Observed: {observed_modularity:.3f}, p={p_value:.3f}")
    return output


if __name__ == "__main__":
    calculate_rates()
    calculate_climate_correlation()
    calculate_null_model()
