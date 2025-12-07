import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_diversity_curve(data_path="data/processed/occurrences.parquet", output_dir="data/analysis"):
    """
    Plots a simple diversity curve (number of genera per time bin).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Filter for valid time data
    df = df.dropna(subset=["mid_ma", "genus"])

    # Create 5my bins
    # We'll just round mid_ma to nearest 5
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5

    # Count unique genera per bin
    diversity = df.groupby("time_bin")["genus"].nunique()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(diversity.index, diversity.values, marker='o')
    plt.gca().invert_xaxis() # Time goes backwards
    plt.xlabel("Time (Ma)")
    plt.ylabel("Number of Genera")
    plt.title("Diversity Curve (Genera)")
    plt.grid(True)
    
    output_file = os.path.join(output_dir, "diversity_curve.png")
    plt.savefig(output_file)
    print(f"Diversity curve saved to {output_file}")

def plot_map(data_path="data/processed/occurrences.parquet", output_dir="data/analysis"):
    """
    Plots a simple map of occurrences.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    df = df.dropna(subset=["lat", "lng"])

    plt.figure(figsize=(12, 6))
    plt.scatter(df["lng"], df["lat"], s=1, alpha=0.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Global Occurrence Map")
    plt.grid(True)

    output_file = os.path.join(output_dir, "occurrence_map.png")
    plt.savefig(output_file)
    print(f"Map saved to {output_file}")
