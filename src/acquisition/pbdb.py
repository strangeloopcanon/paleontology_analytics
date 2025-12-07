import requests
import pandas as pd
import os
import time
from datetime import datetime

PBDB_API_URL = "https://paleobiodb.org/data1.2/occs/list.csv"

def fetch_pbdb_occurrences(
    interval="Cambrian,Cretaceous",
    output_dir="data/raw",
    filename=None
):
    """
    Fetches occurrence data from the Paleobiology Database (PBDB).

    Args:
        interval (str): Time interval to fetch data for (e.g., "Cambrian,Cretaceous").
        output_dir (str): Directory to save the data.
        filename (str, optional): Custom filename. If None, generates one with timestamp.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pbdb_occurrences_{timestamp}.csv"
    
    output_path = os.path.join(output_dir, filename)
    
    print(f"Fetching PBDB data for interval: {interval}...")
    
    # PBDB API parameters
    # 'all' gets a standard set of fields. 
    # 'show=coords,class,paleoloc' adds coordinates, classification, and paleolocation.
    params = {
        "interval": interval,
        "show": "coords,class,paleoloc,strat,time,env,ref",
        "limit": "all", # Get all records
        "vocab": "pbdb" # Use PBDB vocabulary
    }

    try:
        response = requests.get(PBDB_API_URL, params=params, stream=True)
        response.raise_for_status()

        # Save to CSV
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Data saved to {output_path}")
        
        # Verify it's a valid CSV by reading the first few lines
        df = pd.read_csv(output_path, nrows=5)
        print(f"Successfully downloaded. Columns: {list(df.columns)}")
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from PBDB: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    fetch_pbdb_occurrences()
