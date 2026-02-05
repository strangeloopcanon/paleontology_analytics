import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from src.analysis.geo import add_analysis_coordinates, add_binned_locality
from src.analysis.provinciality import compute_locality_network_modularity
from src.analysis.cleaning import clean_taxon_series

def analyze_biogeographic_dynamics(data_path="data/processed/merged_occurrences.parquet", output_dir="data/analysis"):
    """
    Performs 'The Pulse of Pangea' analysis:
    1. Time-series of Network Modularity (Provinciality proxy).
    2. Time-series of Latitudinal Centroids.
    3. Correlation between Modularity and Diversity.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data for SOTA analysis from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Filter valid data
    df["genus"] = clean_taxon_series(df["genus"])
    df = df.dropna(subset=["mid_ma", "genus"])
    df = add_analysis_coordinates(df)
    df = df.dropna(subset=["analysis_lat", "analysis_lng"])
    
    # Create time bins (e.g., 5 Ma)
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    
    # Bin localities for network construction
    df = add_binned_locality(df, bin_degrees=5.0)

    results = []
    
    # Iterate over time bins
    # To ensure robust networks, we need enough data per bin.
    # We'll skip bins with too few occurrences.
    
    for time_bin, group in df.groupby("time_bin"):
        if len(group) < 100: # Minimum occurrences
            continue
            
        # 1. Calculate Modularity (Provinciality proxy)
        res = compute_locality_network_modularity(group, locality_col="locality", genus_col="genus")
        modularity = res.modularity if res.modularity is not None else np.nan

        # 2. Calculate Latitudinal Centroid
        # Weighted by occurrence count? Or just mean of occurrences?
        # Let's do mean absolute latitude of occurrences to see contraction/expansion
        mean_abs_lat = group["analysis_lat"].abs().mean()
        
        # 3. Diversity (Raw Genus Count for simplicity here, or use SQS if integrated)
        diversity = group["genus"].nunique()
        
        results.append({
            "time_bin": time_bin,
            "modularity": modularity,
            "mean_abs_lat": mean_abs_lat,
            "diversity": diversity
        })

    if not results:
        print("Insufficient data for SOTA analysis after filtering (no bins met minimum occurrence threshold).")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("time_bin", ascending=False) # Oldest to Youngest? No, time_bin is Ma, so High to Low.
    
    # Plot 1: Modularity vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["time_bin"], results_df["modularity"], marker='o', label="Network Modularity (Provinciality)")
    plt.gca().invert_xaxis()
    plt.xlabel("Time (Ma)")
    plt.ylabel("Network Modularity")
    plt.title("Evolution of Biogeographic Provinciality (Co-occurrence Network)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "modularity_over_time.png"))
    
    # Plot 2: Latitudinal Centroid vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["time_bin"], results_df["mean_abs_lat"], marker='o', color='green', label="Mean Abs Latitude")
    plt.gca().invert_xaxis()
    plt.xlabel("Time (Ma)")
    plt.ylabel("Mean Absolute Latitude (degrees)")
    plt.title("Latitudinal Shift of Occurrence Distribution")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "latitudinal_shift.png"))
    
    # Plot 3: Modularity vs Diversity Correlation
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df["modularity"], results_df["diversity"])
    plt.xlabel("Modularity")
    plt.ylabel("Diversity (Genus Count)")
    plt.title("Provincialism vs. Diversity")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "modularity_vs_diversity.png"))
    
    print("SOTA analysis complete. Plots saved.")

if __name__ == "__main__":
    analyze_biogeographic_dynamics()
