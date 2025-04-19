import os
import requests
import logging
from typing import Dict, Any, Optional

# Setup logging
log = logging.getLogger(__name__)

# Load API key from environment variables
API_KEY = os.getenv("SANTIMENT_API_KEY")
BASE_URL = "https://api.santiment.net/graphql"

if not API_KEY:
    log.error("Santiment API key is not set. Please add it to your .env file.")


def fetch_social_volume(slug: str, from_date: str, to_date: str, interval: str = "1d") -> Optional[Dict[str, Any]]:
    """
    Fetches social volume data for a given cryptocurrency slug.

    Args:
        slug (str): The cryptocurrency slug (e.g., "ethereum", "bitcoin").
        from_date (str): Start date in ISO 8601 format (e.g., "2023-01-01T00:00:00Z").
        to_date (str): End date in ISO 8601 format (e.g., "2023-01-05T00:00:00Z").
        interval (str): Data interval (e.g., "1d" for daily).

    Returns:
        Optional[Dict[str, Any]]: The social volume data or None if the request fails.
    """
    query = {
        "query": f"""
        {{
            getMetric(metric: "social_volume_total") {{
                timeseriesData(
                    slug: "{slug}",
                    from: "{from_date}",
                    to: "{to_date}",
                    interval: "{interval}"
                ) {{
                    datetime
                    value
                }}
            }}
        }}
        """
    }
    headers = {"Authorization": f"Apikey {API_KEY}"}

    try:
        response = requests.post(BASE_URL, json=query, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch social volume data for {slug}: {e}")
        return None


def fetch_active_addresses(slug: str, from_date: str, to_date: str, interval: str = "1d") -> Optional[Dict[str, Any]]:
    """
    Fetches active addresses data for a given cryptocurrency slug.

    Args:
        slug (str): The cryptocurrency slug (e.g., "ethereum", "bitcoin").
        from_date (str): Start date in ISO 8601 format (e.g., "2023-01-01T00:00:00Z").
        to_date (str): End date in ISO 8601 format (e.g., "2023-01-05T00:00:00Z").
        interval (str): Data interval (e.g., "1d" for daily).

    Returns:
        Optional[Dict[str, Any]]: The active addresses data or None if the request fails.
    """
    query = {
        "query": f"""
        {{
            getMetric(metric: "active_addresses_24h") {{
                timeseriesData(
                    slug: "{slug}",
                    from: "{from_date}",
                    to: "{to_date}",
                    interval: "{interval}"
                ) {{
                    datetime
                    value
                }}
            }}
        }}
        """
    }
    headers = {"Authorization": f"Apikey {API_KEY}"}

    try:
        response = requests.post(BASE_URL, json=query, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch active addresses data for {slug}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fetch social volume data
    social_volume = fetch_social_volume("ethereum", "2023-01-01T00:00:00Z", "2023-01-05T00:00:00Z")
    if social_volume:
        log.info(f"Social volume data: {social_volume}")

    # Fetch active addresses data
    active_addresses = fetch_active_addresses("ethereum", "2023-01-01T00:00:00Z", "2023-01-05T00:00:00Z")
    if active_addresses:
        log.info(f"Active addresses data: {active_addresses}")