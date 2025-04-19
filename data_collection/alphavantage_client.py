"""
AlphaVantage API Client

This module handles interaction with the AlphaVantage API to fetch crypto health metrics.
"""

import os
import sys
import logging
import requests
import time
from typing import Dict, Any, Optional
import datetime
import pytz
import pandas as pd
import aiohttp
import asyncio

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import config
from utils.rate_limiter import AsyncRateLimiter

log = logging.getLogger(__name__)

API_BASE_URL = "https://www.alphavantage.co/query"

# AlphaVantage has a limit of 5 API calls per minute for free tier
# Using AsyncRateLimiter for async operations
rate_limiter = AsyncRateLimiter(calls_per_minute=5)  # 5 calls per minute (free tier limit)

async def fetch_crypto_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch cryptocurrency data from AlphaVantage API.
    
    Args:
        symbol: The cryptocurrency symbol to fetch data for
        
    Returns:
        Dictionary containing price and volume data if successful, None otherwise
    """
    if not config.ALPHAVANTAGE_API_KEY:
        log.error("AlphaVantage API key not configured")
        return None
        
    url = config.ALPHAVANTAGE_BASE_URL
    params = {
        'function': 'CRYPTO_INTRADAY',
        'symbol': symbol,
        'market': 'USD',
        'interval': '5min',
        'apikey': config.ALPHAVANTAGE_API_KEY
    }
    
    try:
        # Wait if needed to respect rate limits
        await rate_limiter.wait_if_needed()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Error Message" in data:
                        log.error(f"AlphaVantage API error: {data['Error Message']}")
                        return None
                    return {
                        'symbol': symbol,
                        'data': data
                    }
                else:
                    log.error(f"Error fetching AlphaVantage data: {response.status}")
                    return None
    except Exception as e:
        log.error(f"Exception in AlphaVantage fetch: {e}")
        return None

async def fetch_technical_indicators(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch technical indicators from AlphaVantage API.
    
    Args:
        symbol: The cryptocurrency symbol to fetch indicators for
        
    Returns:
        Dictionary containing technical indicators if successful, None otherwise
    """
    if not config.ALPHAVANTAGE_API_KEY:
        log.error("AlphaVantage API key not configured")
        return None
        
    indicators = {}
    
    try:
        # Fetch RSI
        await rate_limiter.wait_if_needed()
        async with aiohttp.ClientSession() as session:
            params = {
                'function': 'RSI',
                'symbol': f"CRYPTO:{symbol}",
                'interval': 'daily',
                'time_period': '14',
                'series_type': 'close',
                'apikey': config.ALPHAVANTAGE_API_KEY
            }
            async with session.get(config.ALPHAVANTAGE_BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Technical Analysis: RSI" in data:
                        indicators['RSI'] = data["Technical Analysis: RSI"]
        
        # Fetch MACD
        await rate_limiter.wait_if_needed()
        async with aiohttp.ClientSession() as session:
            params = {
                'function': 'MACD',
                'symbol': f"CRYPTO:{symbol}",
                'interval': 'daily',
                'series_type': 'close',
                'apikey': config.ALPHAVANTAGE_API_KEY
            }
            async with session.get(config.ALPHAVANTAGE_BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Technical Analysis: MACD" in data:
                        indicators['MACD'] = data["Technical Analysis: MACD"]
        
        return indicators if indicators else None
        
    except Exception as e:
        log.error(f"Exception in AlphaVantage technical indicators fetch: {e}")
        return None

async def fetch_crypto_health_index(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch crypto health index data from AlphaVantage API.
    
    Args:
        symbol: Cryptocurrency symbol to fetch data for
        
    Returns:
        Dictionary containing health index data if successful, None otherwise
    """
    if not config.ALPHAVANTAGE_API_KEY:
        log.error("AlphaVantage API key not configured")
        return None
        
    try:
        await rate_limiter.wait_if_needed()
        params = {
            'function': 'CRYPTO_RATING',
            'symbol': symbol,
            'apikey': config.ALPHAVANTAGE_API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(API_BASE_URL, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    log.error(f"Error fetching AlphaVantage data: {response.status}")
                    return None
    except Exception as e:
        log.error(f"Exception in AlphaVantage API call: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fetch health index
    loop = asyncio.get_event_loop()
    health_data = loop.run_until_complete(fetch_crypto_health_index("BTC"))
    if health_data:
        log.info(f"Health index data: {health_data}")