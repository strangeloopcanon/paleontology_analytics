import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_biogeographic_network(data_path="data/processed/merged_occurrences.parquet", output_dir="data/analysis"):
    """
    Constructs and plots a biogeographic network.
    Nodes = Localities (lat/lng bins), Edges = Shared Taxa.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data for network analysis from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Filter valid data
    df = df.dropna(subset=["lat", "lng", "genus"])
    
    # Bin localities (e.g., 5x5 degree grid) to reduce sparsity
    df["lat_bin"] = (df["lat"] / 5).round() * 5
    df["lng_bin"] = (df["lng"] / 5).round() * 5
    df["locality"] = list(zip(df["lat_bin"], df["lng_bin"]))

    # Create Locality-by-Taxon Matrix
    # We want to see which localities share genera
    # This can be huge, so let's filter for top genera or just do it for a specific time slice if needed.
    # For this demo, we'll take a sample or a specific period if column exists.
    
    # Let's filter for a specific period to make it meaningful, e.g., Jurassic if available, or just all.
    # To avoid OOM on huge datasets, let's limit to top 50 localities by occurrence count.
    top_localities = df["locality"].value_counts().head(50).index
    df_filtered = df[df["locality"].isin(top_localities)]
    
    # Create bipartite graph or projection
    # Locality -> Genus
    G = nx.Graph()
    
    # Add nodes
    localities = df_filtered["locality"].unique()
    genera = df_filtered["genus"].unique()
    
    G.add_nodes_from(localities, bipartite=0)
    G.add_nodes_from(genera, bipartite=1)
    
    # Add edges
    edges = list(zip(df_filtered["locality"], df_filtered["genus"]))
    G.add_edges_from(edges)
    
    # Project to Locality-Locality network
    locality_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    locality_graph = nx.bipartite.projected_graph(G, locality_nodes)
    
    # Plot
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(locality_graph, k=0.15, iterations=20)
    nx.draw_networkx_nodes(locality_graph, pos, node_size=50, node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(locality_graph, pos, alpha=0.1)
    plt.title("Biogeographic Network (Shared Genera between Localities)")
    plt.axis('off')
    
    output_file = os.path.join(output_dir, "biogeographic_network.png")
    plt.savefig(output_file)
    print(f"Network graph saved to {output_file}")

def calculate_sqs_diversity(data_path="data/processed/merged_occurrences.parquet", output_dir="data/analysis", quota=0.5):
    """
    Calculates Shareholder Quorum Subsampling (SQS) diversity.
    This is a simplified implementation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data for SQS from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    df = df.dropna(subset=["mid_ma", "genus"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    
    # Group by time bin
    bins = df["time_bin"].unique()
    sqs_results = {}
    
    for bin_val in sorted(bins):
        bin_data = df[df["time_bin"] == bin_val]
        counts = bin_data["genus"].value_counts()
        
        # SQS Logic (Simplified)
        # Sort genera by frequency
        # Sum frequencies until quota is reached
        # Count how many genera that is
        
        total_occurrences = counts.sum()
        if total_occurrences == 0:
            continue
            
        freqs = counts / total_occurrences
        
        # Sort descending
        freqs = freqs.sort_values(ascending=False)
        
        cumulative_freq = 0
        genera_count = 0
        for f in freqs:
            cumulative_freq += f
            genera_count += 1
            if cumulative_freq >= quota:
                break
        
        sqs_results[bin_val] = genera_count

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(sqs_results.keys()), list(sqs_results.values()), marker='o', color='orange')
    plt.gca().invert_xaxis()
    plt.xlabel("Time (Ma)")
    plt.ylabel(f"SQS Diversity (Quota={quota})")
    plt.title("Subsampled Diversity Curve (SQS)")
    plt.grid(True)
    
    output_file = os.path.join(output_dir, "sqs_diversity.png")
    plt.savefig(output_file)
    print(f"SQS curve saved to {output_file}")
