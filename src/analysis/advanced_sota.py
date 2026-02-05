import pandas as pd
import numpy as np
import json

from src.analysis.geo import add_analysis_coordinates, add_binned_locality
from src.analysis.provinciality import compute_locality_network_modularity
from src.analysis.cleaning import clean_taxon_series

def calculate_rates(data_path="data/processed/merged_occurrences.parquet", output_file="dashboard/rates_data.json"):
    """
    Calculate origination and extinction rates per time bin.
    
    - Origination Rate = (genera with first appearance in bin) / (total genera observed in bin)
    - Extinction Rate = (genera present in bin that are absent in the next, younger bin) / (total genera observed in bin)

    Notes:
    - These are dataset-based proxies and can be influenced by sampling and binning choices.
    - The youngest bin is right-censored (no younger bin to compare to), so extinction is reported as null there.
    """
    print("Calculating origination/extinction rates...")
    
    df = pd.read_parquet(data_path)
    df["genus"] = clean_taxon_series(df["genus"])
    df = df.dropna(subset=["mid_ma", "genus"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5

    time_bins = sorted(df["time_bin"].unique(), reverse=True)  # Oldest first

    first_bin = df.groupby("genus")["time_bin"].max()

    results = []
    for i, current_bin in enumerate(time_bins):
        current_genera = set(df[df["time_bin"] == current_bin]["genus"].unique())
        total = int(len(current_genera))
        if total == 0:
            continue

        originations = int((first_bin == current_bin).sum())
        orig_rate = originations / total if total else None

        if i < len(time_bins) - 1:
            next_bin = time_bins[i + 1]
            next_genera = set(df[df["time_bin"] == next_bin]["genus"].unique())
            extinctions = int(len(current_genera - next_genera))
            ext_rate = extinctions / total if total else None
        else:
            # Youngest bin is right-censored in the dataset.
            extinctions = None
            ext_rate = None

        results.append(
            {
                "time": float(current_bin),
                "origination_rate": orig_rate,
                "extinction_rate": ext_rate,
                "total_genera": total,
                "originations": originations,
                "extinctions": extinctions,
            }
        )
    
    # Detect mass extinctions (extinction rate > 2 std above mean)
    ext_rates = [r["extinction_rate"] for r in results if r["extinction_rate"] is not None]
    mean_ext = np.mean(ext_rates)
    std_ext = np.std(ext_rates)
    threshold = mean_ext + 2 * std_ext
    
    for r in results:
        r["is_mass_extinction"] = bool(r["extinction_rate"] is not None and r["extinction_rate"] > threshold)
    
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
    df["genus"] = clean_taxon_series(df["genus"])
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
        if age <= sorted_times[0]:
            return temp_points[sorted_times[0]]
        if age >= sorted_times[-1]:
            return temp_points[sorted_times[-1]]
        
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


def calculate_null_model(
    data_path="data/processed/merged_occurrences.parquet",
    output_file="dashboard/null_model_data.json",
    n_iterations=100,
    *,
    random_seed: int = 42,
):
    """
    Generate a null distribution for provinciality (network modularity) to test against random mixing.
    """
    print(f"Running null model test ({n_iterations} iterations)...")
    
    df = pd.read_parquet(data_path)
    df["genus"] = clean_taxon_series(df["genus"])
    df = df.dropna(subset=["mid_ma", "genus"])
    df = add_analysis_coordinates(df)
    df = df.dropna(subset=["analysis_lat", "analysis_lng"])
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    df = add_binned_locality(df, bin_degrees=5.0, locality_col="locality")
    
    # Pick a representative time bin with good data
    bin_sizes = df.groupby("time_bin").size()
    target_bin = bin_sizes.idxmax()  # Use the bin with most data
    
    group = df[df["time_bin"] == target_bin]
    if group["locality"].nunique() < 10 or group["genus"].nunique() < 10:
        print("Insufficient data for null model test")
        return

    edges_df = group[["locality", "genus"]].dropna().drop_duplicates().reset_index(drop=True)
    observed_res = compute_locality_network_modularity(edges_df, locality_col="locality", genus_col="genus", min_localities=10, min_genera=10)
    observed_modularity = observed_res.modularity

    # Null distribution: shuffle genus labels across locality–genus edges (keeps locality degree + genus frequency).
    null_modularities = []
    rng = np.random.default_rng(random_seed)

    for _ in range(n_iterations):
        shuffled_edges_df = edges_df.copy()
        shuffled_edges_df["genus"] = rng.permutation(shuffled_edges_df["genus"].values)
        null_res = compute_locality_network_modularity(
            shuffled_edges_df, locality_col="locality", genus_col="genus", min_localities=10, min_genera=10
        )
        if null_res.modularity is not None:
            null_modularities.append(float(null_res.modularity))
    
    # Calculate p-value
    if observed_modularity is None or len(null_modularities) == 0:
        p_value = 1.0
        k_ge = 0
    else:
        k_ge = sum(1 for m in null_modularities if m >= observed_modularity)
        # One-sided permutation p-value with +1 correction so p is never exactly 0.
        p_value = (k_ge + 1) / (len(null_modularities) + 1)
    
    output = {
        "observed_modularity": observed_modularity,
        "null_distribution": null_modularities,
        "p_value": p_value,
        "p_value_k_ge_observed": int(k_ge),
        "n_iterations_requested": int(n_iterations),
        "n_iterations_used": int(len(null_modularities)),
        "random_seed": int(random_seed),
        "time_bin": float(target_bin),
        "significant": bool(p_value < 0.05),
        "n_edges": int(len(edges_df)),
        "n_localities": int(edges_df["locality"].nunique()),
        "n_genera": int(edges_df["genus"].nunique()),
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f)
    
    observed_text = f"{observed_modularity:.3f}" if observed_modularity is not None else "NA"
    print(f"Null model saved. Observed: {observed_text}, p={p_value:.3f}")
    return output


if __name__ == "__main__":
    calculate_rates()
    calculate_climate_correlation()
    calculate_null_model()
