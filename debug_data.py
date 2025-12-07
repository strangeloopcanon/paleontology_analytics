import pandas as pd
import os

raw_file = "data/raw/pbdb_occurrences_20251204_001006.csv"

print(f"Reading {raw_file}...")
try:
    # Read only necessary columns to save memory/time
    df = pd.read_csv(raw_file, usecols=["occurrence_no", "max_ma", "min_ma", "phylum"])
    print(f"Total rows read: {len(df)}")
    
    # Check max_ma type
    print(f"max_ma dtype: {df['max_ma'].dtype}")
    
    # Convert to numeric
    df["max_ma_num"] = pd.to_numeric(df["max_ma"], errors='coerce')
    
    print(f"Rows with valid max_ma: {df['max_ma_num'].notna().sum()}")
    print(f"Max age found: {df['max_ma_num'].max()} Ma")
    print(f"Min age found: {df['max_ma_num'].min()} Ma")
    
    # Check distribution
    print("\nAge distribution (bins):")
    print(pd.cut(df["max_ma_num"], bins=[0, 66, 252, 541], labels=["Cenozoic", "Mesozoic", "Paleozoic"]).value_counts())

except Exception as e:
    print(f"Error: {e}")
