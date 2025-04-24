import os
import logging
from typing import Dict, Any, Optional
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from database.db_utils import async_bulk_insert

# Setup logging
log = logging.getLogger(__name__)

# Load API key
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
BASE_URL = "https://min-api.cryptocompare.com/data"

if not API_KEY:
    log.error("CryptoCompare API key is not set. Please add it to your .env file.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_price_data(symbol: str, currency: str = "USD") -> Optional[Dict[str, Any]]:
    """Fetch real-time price data from CryptoCompare."""
    url = f"{BASE_URL}/price"
    params = {
        'fsym': symbol,
        'tsyms': currency,
        'api_key': API_KEY
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if currency in data:
                        price_data = {
                            'symbol': symbol,
                            'currency': currency,
                            'price': data[currency],
                            'timestamp': 'now'  # CryptoCompare will use server time
                        }
                        
                        # Store in database
                        await async_bulk_insert([price_data], 'cryptocompare_prices')
                        return price_data
                        
                return None
                
    except Exception as e:
        log.error(f"Failed to fetch price data for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_historical_data(symbol: str, currency: str = "USD", limit: int = 30) -> Optional[Dict[str, Any]]:
    """Fetch historical price data from CryptoCompare."""
    url = f"{BASE_URL}/histoday"
    params = {
        'fsym': symbol,
        'tsym': currency,
        'limit': limit,
        'api_key': API_KEY
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'Data' in data:
                        history = data['Data']
                        processed_data = [{
                            'symbol': symbol,
                            'currency': currency,
                            'timestamp': day['time'],
                            'open': day['open'],
                            'high': day['high'],
                            'low': day['low'],
                            'close': day['close'],
                            'volume': day['volumefrom']
                        } for day in history]
                        
                        # Store in database
                        await async_bulk_insert(processed_data, 'cryptocompare_historical')
                        return {'symbol': symbol, 'data': history}
                        
                return None
                
    except Exception as e:
        log.error(f"Failed to fetch historical data for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_social_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch social media stats from CryptoCompare."""
    url = f"{BASE_URL}/social/coin/latest"
    params = {
        'coinId': symbol,
        'api_key': API_KEY
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'Data' in data:
                        social_stats = data['Data']
                        processed_data = {
                            'symbol': symbol,
                            'reddit_subscribers': social_stats.get('Reddit', {}).get('subscribers', 0),
                            'reddit_active_users': social_stats.get('Reddit', {}).get('active_users', 0),
                            'twitter_followers': social_stats.get('Twitter', {}).get('followers', 0),
                            'twitter_statuses': social_stats.get('Twitter', {}).get('statuses', 0),
                            'github_forks': social_stats.get('Github', {}).get('forks', 0),
                            'github_stars': social_stats.get('Github', {}).get('stars', 0),
                            'timestamp': 'now'
                        }
                        
                        # Store in database
                        await async_bulk_insert([processed_data], 'cryptocompare_social')
                        return processed_data
                        
                return None
                
    except Exception as e:
        log.error(f"Failed to fetch social data for {symbol}: {e}")
        return None