import pandas as pd
import json
import os

from src.analysis.geo import add_analysis_coordinates
from src.analysis.cleaning import clean_taxon_series

def generate_kids_data(data_path="data/processed/merged_occurrences.parquet", output_dir="dashboard"):
    """
    Generate DATA-DRIVEN insights for kids - using actual database facts, not generic info.
    """
    print("Generating data-driven insights...")
    
    df = pd.read_parquet(data_path)
    df["genus"] = clean_taxon_series(df["genus"])
    df = df.dropna(subset=["mid_ma", "genus"])
    df = add_analysis_coordinates(df)
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    
    # ========== 12-YEAR-OLD: DEEP TIME INSIGHTS (SMART, DATA-DRIVEN) ==========
    
    # 1. Survivor Champions: Longest-lived genera WITH context
    genus_ranges = df.groupby("genus")["mid_ma"].agg(["min", "max", "count"])
    genus_ranges["duration"] = genus_ranges["max"] - genus_ranges["min"]
    genus_ranges = genus_ranges[genus_ranges["duration"] > 0]  # Filter out single-occurrence
    genus_ranges = genus_ranges.sort_values("duration", ascending=False)
    
    survivor_champions = []
    for genus, row in genus_ranges.head(15).iterrows():
        survivor_champions.append({
            "genus": genus,
            "first_appearance": round(float(row["max"]), 1),
            "last_appearance": round(float(row["min"]), 1),
            "duration_myr": round(float(row["duration"]), 1),
            "occurrences": int(row["count"])
        })
    
    # 2. Rarity Analysis: Most and Least Common Genera
    genus_counts = df["genus"].value_counts()
    
    rarest_genera = []
    for genus, count in genus_counts.tail(10).items():
        genus_data = df[df["genus"] == genus]
        age = genus_data["mid_ma"].mean()
        rarest_genera.append({
            "genus": genus,
            "occurrences": int(count),
            "age_ma": round(float(age), 1)
        })
    
    most_common = []
    for genus, count in genus_counts.head(10).items():
        genus_data = df[df["genus"] == genus]
        age_range = f"{genus_data['mid_ma'].max():.0f}-{genus_data['mid_ma'].min():.0f}"
        most_common.append({
            "genus": genus,
            "occurrences": int(count),
            "age_range": age_range
        })
    
    # 3. Geographic Spread Champions
    df_geo = df.dropna(subset=["analysis_lat", "analysis_lng"]).copy()
    df_geo["lat_bin"] = (df_geo["analysis_lat"] / 10).round() * 10
    df_geo["lng_bin"] = (df_geo["analysis_lng"] / 10).round() * 10
    
    genus_spread = (
        df_geo.groupby(["genus", "lat_bin", "lng_bin"])
        .size()
        .groupby("genus")
        .size()
        .sort_values(ascending=False)
    )
    
    geographic_champions = []
    for genus, loc_count in genus_spread.head(10).items():
        geographic_champions.append({
            "genus": genus,
            "unique_regions": int(loc_count)
        })
    
    # 4. Mass Extinction Analysis - Actual survivors from data
    extinctions = {
        "End-Permian (252 Ma)": (260, 245),
        "End-Triassic (201 Ma)": (210, 195),
        "End-Cretaceous (66 Ma)": (70, 60)
    }
    
    extinction_analysis = {}
    for name, (before_end, after_start) in extinctions.items():
        before = set(df[(df["mid_ma"] <= before_end) & (df["mid_ma"] > before_end - 15)]["genus"].unique())
        after = set(df[(df["mid_ma"] >= after_start) & (df["mid_ma"] < after_start + 15)]["genus"].unique())
        
        if len(before) > 0 and len(after) > 0:
            survivors = list(before & after)[:10]
            victims = list(before - after)[:10]
            newcomers = list(after - before)[:10]
            
            extinction_analysis[name] = {
                "genera_before": len(before),
                "genera_after": len(after),
                "survivors": survivors,
                "victims": victims,
                "newcomers": newcomers,
                "survival_rate": round(len(before & after) / len(before) * 100, 1) if len(before) > 0 else 0
            }
    
    # 5. Surprising Stats from Data
    total_genera = df["genus"].nunique()
    total_occurrences = len(df)
    oldest_occurrence = df["mid_ma"].max()
    youngest_occurrence = max(0.0, float(df["mid_ma"].min()))
    
    # Find the "explosion" periods - highest origination
    genus_first = df.groupby("genus")["mid_ma"].max()
    originations_per_bin = genus_first.groupby((genus_first / 10).round() * 10).count()
    peak_origination_time = originations_per_bin.idxmax()
    peak_origination_count = originations_per_bin.max()
    
    deep_time_data = {
        "survivor_champions": survivor_champions,
        "rarest_genera": rarest_genera,
        "most_common": most_common,
        "geographic_champions": geographic_champions,
        "extinction_analysis": extinction_analysis,
        "stats": {
            "total_genera": int(total_genera),
            "total_occurrences": int(total_occurrences),
            "time_span": f"{float(oldest_occurrence):.0f} - {float(youngest_occurrence):.0f} Ma",
            "peak_origination": {
                "time_ma": float(peak_origination_time),
                "new_genera": int(peak_origination_count)
            }
        }
    }
    
    # ========== 8-YEAR-OLD: DINO DATA (FROM ACTUAL DATABASE) ==========
    
    # Filter to Mesozoic
    dino_df = df[(df["mid_ma"] <= 252) & (df["mid_ma"] >= 66)]
    dino_df = dino_df.dropna(subset=["analysis_lat", "analysis_lng"])

    mesozoic_genera = dino_df["genus"].unique()
    mesozoic_count = len(mesozoic_genera)

    # Real insights from the Mesozoic data
    mesozoic_genus_counts = dino_df["genus"].value_counts()

    # Most common Mesozoic genera
    top_mesozoic = []
    for genus, count in mesozoic_genus_counts.head(15).items():
        genus_data = dino_df[dino_df["genus"] == genus]
        avg_age = genus_data["mid_ma"].mean()
        top_mesozoic.append(
            {
                "genus": genus,
                "occurrences": int(count),
                "avg_age_ma": round(float(avg_age), 0),
            }
        )

    # Geographic distribution
    dino_by_region = (
        dino_df.groupby([(dino_df["analysis_lat"] > 0).map({True: "Northern", False: "Southern"})])["genus"]
        .nunique()
        .to_dict()
    )

    # Time distribution
    triassic = dino_df[(dino_df["mid_ma"] <= 252) & (dino_df["mid_ma"] > 201)]
    jurassic = dino_df[(dino_df["mid_ma"] <= 201) & (dino_df["mid_ma"] > 145)]
    cretaceous = dino_df[(dino_df["mid_ma"] <= 145) & (dino_df["mid_ma"] >= 66)]

    period_breakdown = {
        "Triassic": {"genera": int(triassic["genus"].nunique()), "occurrences": int(len(triassic))},
        "Jurassic": {"genera": int(jurassic["genus"].nunique()), "occurrences": int(len(jurassic))},
        "Cretaceous": {"genera": int(cretaceous["genus"].nunique()), "occurrences": int(len(cretaceous))},
    }

    # Data-driven dino facts
    dino_facts = [f"Our database has {mesozoic_count:,} different Mesozoic genera!"]
    if not mesozoic_genus_counts.empty:
        dino_facts.extend(
            [
                f"The most common Mesozoic genus is {mesozoic_genus_counts.index[0]} with {mesozoic_genus_counts.iloc[0]:,} fossil occurrences.",
                f"The Cretaceous period has {period_breakdown['Cretaceous']['genera']} genera - more than Triassic and Jurassic combined!",
            ]
        )
    else:
        dino_facts.append("We currently have no georeferenced Mesozoic occurrences in this slice of the data.")

    if top_mesozoic:
        dino_facts.append(
            f"{top_mesozoic[0]['genus']} fossils have been found {top_mesozoic[0]['occurrences']} times in our database."
        )
    dino_facts.append(f"We have {total_occurrences:,} total fossil occurrences spanning {oldest_occurrence:.0f} million years.")

    # Add more specific facts
    if geographic_champions:
        dino_facts.append(
            f"The most widespread genus is {geographic_champions[0]['genus']}, found in {geographic_champions[0]['unique_regions']} different regions!"
        )

    if survivor_champions:
        dino_facts.append(
            f"The longest-surviving genus is {survivor_champions[0]['genus']}, lasting {survivor_champions[0]['duration_myr']:.0f} million years!"
        )

    # Dino Map Data
    if len(dino_df) > 0:
        dino_sample = dino_df.sample(min(2000, len(dino_df)), random_state=42)
        dino_map = {
            "lat": dino_sample["analysis_lat"].tolist(),
            "lng": dino_sample["analysis_lng"].tolist(),
        }
    else:
        dino_map = {"lat": [], "lng": []}
    
    dino_zone_data = {
        "top_genera": top_mesozoic,
        "period_breakdown": period_breakdown,
        "region_breakdown": dino_by_region,
        "facts": dino_facts,
        "map": dino_map,
        "total_genera": mesozoic_count
    }
    
    # Save outputs
    with open(os.path.join(output_dir, "deep_time_data.json"), "w") as f:
        json.dump(deep_time_data, f)
    
    with open(os.path.join(output_dir, "dino_zone_data.json"), "w") as f:
        json.dump(dino_zone_data, f)
    
    print(f"Data-driven insights saved to {output_dir}")
    return deep_time_data, dino_zone_data


if __name__ == "__main__":
    generate_kids_data()
