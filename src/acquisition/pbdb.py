import os
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src._logging import get_logger

logger = get_logger(__name__)

PBDB_API_URL = "https://paleobiodb.org/data1.2/occs/list.csv"


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
    
    logger.info("fetching_pbdb_data", extra={"interval": interval, "url": PBDB_API_URL})

    params = {
        "interval": interval,
        "show": "coords,class,paleoloc,strat,time,env,ref",
        "limit": "all",
        "vocab": "pbdb",
    }

    try:
        with _build_retry_session() as session:
            response = session.get(PBDB_API_URL, params=params, stream=True, timeout=60)
            response.raise_for_status()

            # Save to CSV
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logger.info(
            "data_saved",
            extra={"path": output_path, "status_code": response.status_code},
        )

        df = pd.read_csv(output_path, nrows=5)
        logger.info(
            "download_verified",
            extra={"path": output_path, "columns": list(df.columns)},
        )
        return output_path

    except requests.exceptions.RequestException as e:
        logger.error("pbdb_fetch_error", extra={"error": str(e), "url": PBDB_API_URL})
        return None

if __name__ == "__main__":
    # Example usage
    fetch_pbdb_occurrences()
