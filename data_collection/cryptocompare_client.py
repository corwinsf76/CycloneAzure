import os
import requests
import logging
from typing import Dict, Any, Optional

# Setup logging
log = logging.getLogger(__name__)

# Load API key from environment variables
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
BASE_URL = "https://min-api.cryptocompare.com/data"

if not API_KEY:
    log.error("CryptoCompare API key is not set. Please add it to your .env file.")


def fetch_price_data(symbol: str, currency: str = "USD") -> Optional[Dict[str, Any]]:
    """
    Fetches real-time price data for a given cryptocurrency symbol.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., BTC, ETH).
        currency (str): The fiat currency to convert to (default is USD).

    Returns:
        Optional[Dict[str, Any]]: The price data or None if the request fails.
    """
    endpoint = f"{BASE_URL}/price"
    params = {
        "fsym": symbol,
        "tsyms": currency,
        "api_key": API_KEY
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch price data for {symbol}: {e}")
        return None


def fetch_historical_data(symbol: str, currency: str = "USD", limit: int = 30) -> Optional[Dict[str, Any]]:
    """
    Fetches historical price data for a given cryptocurrency symbol.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., BTC, ETH).
        currency (str): The fiat currency to convert to (default is USD).
        limit (int): The number of historical data points to fetch (default is 30).

    Returns:
        Optional[Dict[str, Any]]: The historical price data or None if the request fails.
    """
    endpoint = f"{BASE_URL}/v2/histoday"
    params = {
        "fsym": symbol,
        "tsym": currency,
        "limit": limit,
        "api_key": API_KEY
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch historical data for {symbol}: {e}")
        return None


def fetch_social_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches social sentiment data for a given cryptocurrency symbol.

    Args:
        symbol (str): The cryptocurrency symbol (e.g., BTC, ETH).

    Returns:
        Optional[Dict[str, Any]]: The social sentiment data or None if the request fails.
    """
    endpoint = f"{BASE_URL}/social/coin/latest"
    params = {
        "coinId": symbol,
        "api_key": API_KEY
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        log.error(f"Failed to fetch social data for {symbol}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fetch real-time price data
    price_data = fetch_price_data("BTC")
    if price_data:
        log.info(f"Real-time price data: {price_data}")

    # Fetch historical price data
    historical_data = fetch_historical_data("BTC")
    if historical_data:
        log.info(f"Historical price data: {historical_data}")

    # Fetch social sentiment data
    social_data = fetch_social_data("BTC")
    if social_data:
        log.info(f"Social sentiment data: {social_data}")