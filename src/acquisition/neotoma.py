import requests
import os
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NEOTOMA_API_URL = "https://api.neotomadb.org/v2.0/data/occurrences"


def _build_retry_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_neotoma_data(
    limit=10000,
    output_dir="data/raw",
    filename=None
):
    """
    Fetches occurrence data from Neotoma.
    
    Args:
        limit (int): Max number of records to fetch (Neotoma API pagination might be needed for huge datasets).
        output_dir (str): Directory to save the data.
        filename (str, optional): Custom filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neotoma_occurrences_{timestamp}.json"
    
    output_path = os.path.join(output_dir, filename)
    
    print(f"Fetching Neotoma data (limit={limit})...")
    
    params = {
        "limit": limit,
        "offset": 0
    }

    try:
        # Neotoma API returns JSON
        with _build_retry_session() as session:
            response = session.get(NEOTOMA_API_URL, params=params, timeout=60)
            response.raise_for_status()
        
            data = response.json()
        
        # The data is usually under 'data' key
        if 'data' in data:
            occurrences = data['data']
        else:
            print("Unexpected Neotoma API response structure.")
            return None

        # Save as JSON (preserving structure) or CSV? 
        # JSON is safer for nested data, but we'll try to flatten later.
        # Let's save raw JSON for now.
        import json
        with open(output_path, 'w') as f:
            json.dump(data, f)
            
        print(f"Data saved to {output_path}")
        print(f"Fetched {len(occurrences)} records.")
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Neotoma: {e}")
        return None

if __name__ == "__main__":
    fetch_neotoma_data(limit=100)
