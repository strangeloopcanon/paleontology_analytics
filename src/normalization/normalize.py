import pandas as pd
import os
import glob
import json
from src.normalization.schema import OCCURRENCE_SCHEMA, PBDB_MAPPING

def normalize_pbdb(input_dir="data/raw", output_dir="data/processed"):
    """
    Normalizes PBDB data to the canonical schema.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(os.path.join(input_dir, "pbdb_occurrences_*.csv"))
    if not files:
        print("No PBDB data found in", input_dir)
        return None

    print(f"Found {len(files)} PBDB files. Merging...")
    
    dfs = []
    for f in files:
        print(f"Reading {f}...")
        try:
            # Read CSV
            # Use low_memory=False to avoid mixed type warnings on large files
            df_chunk = pd.read_csv(f, low_memory=False)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return None
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merge: {len(df)}")

    df = df.rename(columns=PBDB_MAPPING)
    df["source_db"] = "PBDB"

    if "mid_ma" not in df.columns and "max_ma" in df.columns:
        # Ensure numeric
        df["max_ma"] = pd.to_numeric(df["max_ma"], errors='coerce')
        df["min_ma"] = pd.to_numeric(df["min_ma"], errors='coerce')
        df["mid_ma"] = (df["max_ma"] + df["min_ma"]) / 2

    return _finalize_dataframe(df, output_dir, "pbdb_occurrences.parquet")

def normalize_neotoma(input_dir="data/raw", output_dir="data/processed"):
    """
    Normalizes Neotoma data to the canonical schema.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(os.path.join(input_dir, "neotoma_occurrences_*.json"))
    if not files:
        print("No Neotoma data found in", input_dir)
        return None

    latest_file = max(files, key=os.path.getctime)
    print(f"Processing Neotoma file: {latest_file}...")

    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        if 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data) # Fallback
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Neotoma Mapping (approximate, needs inspection of real data)
    # Assuming standard Neotoma API response structure
    # We might need to flatten 'site', 'sample', 'taxon' fields if they are nested.
    # For now, let's assume a flattened structure or simple mapping.
    
    # NOTE: Neotoma JSON is often nested. 
    # We'll do a quick check and flatten if needed in a real scenario.
    # For this MVP, we'll map what we can find.
    
    neotoma_mapping = {
        "occurrenceid": "occurrence_id",
        "taxonname": "scientific_name",
        "age": "mid_ma", # Neotoma often has 'age'
        # Coordinates might be in 'site' object
    }
    
    # Flattening logic (simplified)
    if 'site' in df.columns:
        # This is complex without seeing the exact JSON structure.
        # Let's try to extract lat/long if they exist in a 'site' dict
        try:
            df['lat'] = df['site'].apply(lambda x: x.get('geography', {}).get('coordinates', [None, None])[1] if isinstance(x, dict) else None)
            df['lng'] = df['site'].apply(lambda x: x.get('geography', {}).get('coordinates', [None, None])[0] if isinstance(x, dict) else None)
        except:
            pass

    df = df.rename(columns=neotoma_mapping)
    df["source_db"] = "Neotoma"
    
    # Convert age to Ma (Neotoma is often BP)
    if "mid_ma" in df.columns:
        # Ensure mid_ma is numeric, coerce errors to NaN
        # If it's a dict, to_numeric might fail or return NaN depending on how it's passed.
        # Let's force it to string first if it's object, then extract if needed, or just coerce.
        # But if it is a dict, to_numeric won't work directly on the series of dicts.
        
        # Check if it's object type and potentially dicts
        if df["mid_ma"].dtype == 'object':
             # Try to extract 'age' key if it exists in dict
             df["mid_ma"] = df["mid_ma"].apply(lambda x: x.get('age') if isinstance(x, dict) else x)

        df["mid_ma"] = pd.to_numeric(df["mid_ma"], errors='coerce')
        df["mid_ma"] = df["mid_ma"] / 1_000_000

    return _finalize_dataframe(df, output_dir, "neotoma_occurrences.parquet")

def _finalize_dataframe(df, output_dir, filename):
    for col in OCCURRENCE_SCHEMA.keys():
        if col not in df.columns:
            df[col] = None

    df = df[list(OCCURRENCE_SCHEMA.keys())]

    for col, dtype in OCCURRENCE_SCHEMA.items():
        if dtype == "string":
            df[col] = df[col].astype(str)
        elif dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors='coerce')

    output_path = os.path.join(output_dir, filename)
    df.to_parquet(output_path, index=False)
    print(f"Normalized data saved to {output_path}")
    return output_path

def merge_datasets(input_dir="data/processed", output_dir="data/processed"):
    """
    Merges PBDB and Neotoma parquet files.
    """
    files = glob.glob(os.path.join(input_dir, "*_occurrences.parquet"))
    dfs = []
    for f in files:
        if "merged" not in f: # Avoid recursive reading
            try:
                dfs.append(pd.read_parquet(f))
            except:
                pass
    
    if not dfs:
        print("No processed data found to merge.")
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(output_dir, "merged_occurrences.parquet")
    merged_df.to_parquet(output_path, index=False)
    print(f"Merged {len(dfs)} datasets into {output_path}")
    return output_path
