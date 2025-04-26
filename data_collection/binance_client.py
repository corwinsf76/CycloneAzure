# data_collection/binance_client.py

import os
import time
import hmac
import hashlib
import requests

# Load environment variables
BASE_URL = "https://api.binance.us"
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

def _get_signature(query_string):
    """
    Generate HMAC SHA256 signature for Binance API.
    """
    return hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def _send_request(http_method, endpoint, params=None, signed=False):
    """
    Send HTTP request to Binance API.
    
    Args:
        http_method (str): "GET" or "POST"
        endpoint (str): API endpoint starting with "/"
        params (dict): Query parameters
        signed (bool): Whether the request requires signing
    
    Returns:
        dict or list: API response
    """
    if params is None:
        params = {}

    headers = {
        "X-MBX-APIKEY": API_KEY
    }

    if signed:
        params['timestamp'] = int(time.time() * 1000)
        query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])
        params['signature'] = _get_signature(query_string)

    url = f"{BASE_URL}{endpoint}"

    if http_method == "GET":
        response = requests.get(url, headers=headers, params=params)
    elif http_method == "POST":
        response = requests.post(url, headers=headers, data=params)
    else:
        raise ValueError(f"Invalid HTTP method: {http_method}")

    response.raise_for_status()
    return response.json()

def get_binance_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Fetch kline (candlestick) data for a given symbol and interval.

    Args:
        symbol (str): Trading pair symbol, e.g., "BTCUSDT"
        interval (str): Kline interval, e.g., "1m", "5m", "1d"
        start_time (int, optional): Start time in milliseconds
        end_time (int, optional): End time in milliseconds
        limit (int, optional): Number of klines to fetch (default 1000)

    Returns:
        list: List of klines
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    return _send_request("GET", "/api/v3/klines", params=params, signed=False)

# Example future methods you could easily add:

# def get_account_info():
#     return _send_request("GET", "/api/v3/account", signed=True)

# def create_order(symbol, side, type, quantity, price=None):
#     params = {
#         "symbol": symbol,
#         "side": side,
#         "type": type,
#         "quantity": quantity
#     }
#     if price:
#         params["price"] = price
#     return _send_request("POST", "/api/v3/order", params=params, signed=True)
