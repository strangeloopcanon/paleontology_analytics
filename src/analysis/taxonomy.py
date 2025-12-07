import pandas as pd
import numpy as np
import json
import os

def generate_taxonomy_data(data_path="data/processed/merged_occurrences.parquet", output_dir="dashboard"):
    """
    Generate data for:
    1. Sunburst Chart (Phylum -> Class -> Order)
    2. Dinosaur Analysis (Specific stats for Dinosauria)
    """
    print("Generating taxonomy data...")
    
    df = pd.read_parquet(data_path)
    # Ensure columns exist
    for col in ["phylum", "class", "order", "family", "genus"]:
        if col not in df.columns:
            df[col] = "Unknown"
    
    df = df.fillna("Unknown")
    
    # --- 1. Sunburst Data (Phylum -> Class -> Order) ---
    # We'll take the top 50 Orders by occurrence count to keep the chart readable
    top_orders = df["order"].value_counts().head(50).index
    sunburst_df = df[df["order"].isin(top_orders)]
    
    # Group by hierarchy
    hierarchy = sunburst_df.groupby(["phylum", "class", "order"]).size().reset_index(name="count")
    
    # Format for Plotly Sunburst: ids, labels, parents, values
    ids = []
    labels = []
    parents = []
    values = []
    
    # Level 1: Phylum
    phyla = hierarchy.groupby("phylum")["count"].sum()
    for p, v in phyla.items():
        ids.append(p)
        labels.append(p)
        parents.append("")
        values.append(int(v))
        
    # Level 2: Class
    classes = hierarchy.groupby(["phylum", "class"])["count"].sum().reset_index()
    for _, row in classes.iterrows():
        p, c, v = row["phylum"], row["class"], row["count"]
        cid = f"{p}-{c}"
        ids.append(cid)
        labels.append(c)
        parents.append(p)
        values.append(int(v))
        
    # Level 3: Order
    for _, row in hierarchy.iterrows():
        p, c, o, v = row["phylum"], row["class"], row["order"], row["count"]
        oid = f"{p}-{c}-{o}"
        ids.append(oid)
        labels.append(o)
        parents.append(f"{p}-{c}")
        values.append(int(v))
        
    sunburst_data = {
        "ids": ids,
        "labels": labels,
        "parents": parents,
        "values": values
    }
    
    # --- 2. Dinosaur Analysis ---
    # Filter for Dinosauria (Saurischia/Ornithischia) in Class OR Order
    dino_terms = ["Saurischia", "Ornithischia", "Dinosauria"]
    dinos = df[
        (df["class"].isin(dino_terms)) | 
        (df["order"].isin(dino_terms)) |
        (df["phylum"].isin(dino_terms))
    ].copy()
    
    # Ensure numeric age
    dinos["mid_ma"] = pd.to_numeric(dinos["mid_ma"], errors="coerce")
    dinos = dinos.dropna(subset=["mid_ma"])
    
    dino_stats = {
        "total_genera": int(dinos["genus"].nunique()),
        "total_occurrences": int(len(dinos)),
        "time_range": f"{dinos['max_ma'].max():.0f} - {dinos['min_ma'].min():.0f} Ma",
        "top_genera": dinos["genus"].value_counts().head(10).to_dict()
    }
    
    # Diversity over time for Dinos
    dinos["time_bin"] = (dinos["mid_ma"] / 5).round() * 5
    dino_div = dinos.groupby("time_bin")["genus"].nunique().sort_index(ascending=False)
    
    dino_chart = {
        "time": [float(t) for t in dino_div.index],
        "diversity": [int(c) for c in dino_div.values]
    }

    # --- 3. Survivor Champions (Longest Living) ---
    # Calculate duration for ALL genera
    df["mid_ma"] = pd.to_numeric(df["mid_ma"], errors="coerce")
    valid_df = df.dropna(subset=["mid_ma"])
    
    genus_ranges = valid_df.groupby("genus")["mid_ma"].agg(["min", "max", "count"])
    genus_ranges["duration"] = genus_ranges["max"] - genus_ranges["min"]
    # Filter for significant duration (>0) and reasonable occurrence count (>2) to avoid singletons
    survivors = genus_ranges[(genus_ranges["duration"] > 0) & (genus_ranges["count"] > 2)].sort_values("duration", ascending=False).head(15)
    
    survivor_list = []
    for genus, row in survivors.iterrows():
        survivor_list.append({
            "genus": genus,
            "duration": float(row["duration"]),
            "range": f"{row['max']:.0f}-{row['min']:.0f} Ma",
            "occurrences": int(row["count"])
        })

    # Save
    with open(os.path.join(output_dir, "taxonomy_data.json"), "w") as f:
        json.dump({
            "sunburst": sunburst_data,
            "dino_stats": dino_stats,
            "dino_chart": dino_chart,
            "survivors": survivor_list
        }, f)
        
    print(f"Taxonomy data saved to {output_dir}")

if __name__ == "__main__":
    generate_taxonomy_data()
