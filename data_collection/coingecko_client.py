"""
CoinGecko API Client

This module handles interaction with the CoinGecko API to fetch market metrics.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
import datetime
import pytz
import aiohttp
import asyncio
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import config
from utils.rate_limiter import AsyncRateLimiter

log = logging.getLogger(__name__)

API_BASE_URL = "https://api.coingecko.com/api/v3"

# CoinGecko rate limiter - Free tier allows around 10-30 calls per minute
# We'll be conservative with 10 calls per minute to avoid 429 rate limit errors
rate_limiter = AsyncRateLimiter(calls_per_minute=10)

# Cache for coin ID mapping to avoid repeated API calls
_coin_id_cache = {}

async def fetch_coin_metrics(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch coin metrics data from CoinGecko API.
    
    Args:
        symbol: Cryptocurrency symbol to fetch data for
        
    Returns:
        Dictionary containing coin metrics if successful, None otherwise
    """
    try:
        # Respect rate limits
        await rate_limiter.wait_if_needed()
        
        # First get the coin ID from the symbol using the mapping function
        coin_id = await get_coin_id(symbol)
        if not coin_id:
            log.warning(f"Could not find CoinGecko ID for symbol: {symbol}")
            return None
        
        async with aiohttp.ClientSession() as session:
            params = {
                'localization': 'false',
                'tickers': 'false',
                'community_data': 'true',
                'developer_data': 'false',
                'sparkline': 'false'
            }
            
            # Add API key if configured
            if config.COINGECKO_API_KEY:
                headers = {'x-cg-pro-api-key': config.COINGECKO_API_KEY}
            else:
                headers = {}
                
            url = f"{API_BASE_URL}/coins/{coin_id}"
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract relevant metrics
                    metrics = {
                        'symbol': symbol.upper(),
                        'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                        'total_volume': data.get('market_data', {}).get('total_volume', {}).get('usd', 0),
                        'price_change_24h': data.get('market_data', {}).get('price_change_percentage_24h', 0),
                        'market_cap_rank': data.get('market_cap_rank', 0),
                        'community_score': data.get('community_score', 0),
                        'public_interest_score': data.get('public_interest_score', 0),
                        'timestamp': datetime.datetime.now(pytz.utc)
                    }
                    return metrics
                elif response.status == 429:
                    log.error("CoinGecko API rate limit exceeded. Consider reducing request frequency.")
                    # Add a longer delay on rate limit
                    await asyncio.sleep(10)
                    return None
                else:
                    log.error(f"Error fetching CoinGecko data: {response.status}")
                    return None
    except Exception as e:
        log.error(f"Exception in CoinGecko API call: {e}")
        return None

async def fetch_coin_market_data(coin_id: str, vs_currency: str = "usd", days: str = "max", interval: str = "daily") -> Optional[Dict[str, List[float]]]:
    """
    Fetches historical market data for a cryptocurrency.
    
    Args:
        coin_id: CoinGecko coin ID.
        vs_currency: Quote currency (default: usd).
        days: Number of days of data to fetch (default: max).
        interval: Data interval (default: daily).
    
    Returns:
        Dictionary containing price and volume data lists, or None if request fails.
    """
    try:
        # Respect rate limits
        await rate_limiter.wait_if_needed()
        
        async with aiohttp.ClientSession() as session:
            params = {
                "vs_currency": vs_currency,
                "days": days,
                "interval": interval
            }
            
            # Add API key if configured
            if config.COINGECKO_API_KEY:
                headers = {'x-cg-pro-api-key': config.COINGECKO_API_KEY}
            else:
                headers = {}
                
            url = f"{API_BASE_URL}/coins/{coin_id}/market_chart"
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    log.error("CoinGecko API rate limit exceeded. Consider reducing request frequency.")
                    # Add a longer delay on rate limit
                    await asyncio.sleep(10)
                    return None
                else:
                    log.error(f"Error fetching market data for {coin_id}: {response.status}")
                    return None
    
    except Exception as e:
        log.error(f"Unexpected error processing market data for {coin_id}: {e}")
        return None

async def get_coin_id(symbol: str) -> Optional[str]:
    """
    Converts a cryptocurrency symbol to its CoinGecko ID.
    Uses caching to reduce API calls.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC).
    
    Returns:
        CoinGecko coin ID, or None if not found.
    """
    symbol = symbol.lower()
    
    # Check if we have this symbol in cache
    if symbol in _coin_id_cache:
        return _coin_id_cache[symbol]
    
    # Common mappings for popular coins to avoid API calls
    common_mappings = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'usdt': 'tether',
        'bnb': 'binancecoin',
        'sol': 'solana',
        'xrp': 'ripple',
        'ada': 'cardano',
        'doge': 'dogecoin',
        'avax': 'avalanche-2',
        'shib': 'shiba-inu',
        'dot': 'polkadot',
        'link': 'chainlink',
        'matic': 'matic-network',
        'trx': 'tron',
        'uni': 'uniswap',
        'near': 'near',
        'dai': 'dai',
    }
    
    # Check common mappings first
    if symbol in common_mappings:
        _coin_id_cache[symbol] = common_mappings[symbol]
        return common_mappings[symbol]
    
    try:
        # Respect rate limits
        await rate_limiter.wait_if_needed()
        
        # Fetch coin list from API
        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE_URL}/coins/list"
            
            # Add API key if configured
            if config.COINGECKO_API_KEY:
                headers = {'x-cg-pro-api-key': config.COINGECKO_API_KEY}
            else:
                headers = {}
                
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    log.error(f"Error fetching coin list: {response.status}")
                    return None
                
                coins = await response.json()
                
                # Find coin ID by symbol (case-insensitive)
                for coin in coins:
                    if coin.get('symbol', '').lower() == symbol:
                        # Cache the result
                        _coin_id_cache[symbol] = coin['id']
                        return coin['id']
                
                log.warning(f"Symbol {symbol} not found in CoinGecko coin list")
                return None
    
    except Exception as e:
        log.error(f"Error fetching coin list: {e}")
        return None