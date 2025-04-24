"""
CoinGecko API Client

This module handles interaction with the CoinGecko API for cryptocurrency market data.
Uses the Pro plan API with enhanced rate limits.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from database.db_utils import async_bulk_insert
from utils.rate_limiter import AsyncRateLimiter

log = logging.getLogger(__name__)

# API configuration from config
API_BASE_URL = getattr(config, 'COINGECKO_API_BASE', "https://api.coingecko.com/api/v3")
API_CALLS_PER_MINUTE = getattr(config, 'COINGECKO_CALLS_PER_MINUTE', 500)  # Default to Pro plan limit
DEFAULT_CURRENCY = "usd"

# Local cache for coin ID mapping
_coin_id_cache = {}

# Track API usage
_api_call_count = 0

def get_api_headers() -> Dict[str, str]:
    """
    Get headers for CoinGecko API requests, including the Pro API key.
    
    Returns:
        Dictionary of headers
    """
    headers = {
        'accept': 'application/json',
    }
    
    # Add API key if configured
    if config.COINGECKO_API_KEY:
        headers['x-cg-pro-api-key'] = config.COINGECKO_API_KEY
        
    return headers

def track_api_call():
    """Track API call count for monitoring usage"""
    global _api_call_count
    _api_call_count += 1
    
    # Log every 100 calls to help monitor API usage limits
    if _api_call_count % 100 == 0:
        monthly_limit = getattr(config, 'COINGECKO_MONTHLY_LIMIT', 500000)
        log.info(f"CoinGecko API call count: {_api_call_count} (Monthly limit: {monthly_limit})")
        
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_coin_id(symbol: str) -> Optional[str]:
    """
    Get CoinGecko coin ID for a symbol.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'btc', 'eth')
        
    Returns:
        CoinGecko coin ID or None if not found
    """
    symbol = symbol.lower()
    
    # Check cache first
    if symbol in _coin_id_cache:
        return _coin_id_cache[symbol]
    
    # Handle some common mappings manually to avoid API calls
    common_mappings = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'bnb': 'binancecoin',
        'ada': 'cardano',
        'xrp': 'ripple',
        'doge': 'dogecoin',
        'dot': 'polkadot',
        'sol': 'solana',
        'avax': 'avalanche-2',
        'matic': 'polygon',
        'link': 'chainlink',
        'uni': 'uniswap',
        'atom': 'cosmos',
        'near': 'near',
        'algo': 'algorand'
    }
    
    if symbol in common_mappings:
        _coin_id_cache[symbol] = common_mappings[symbol]
        return common_mappings[symbol]
    
    try:
        log.info(f"Fetching coin ID for symbol: {symbol}")
        async with aiohttp.ClientSession() as session:
            headers = get_api_headers()
            async with session.get(f"{API_BASE_URL}/coins/list", headers=headers) as response:
                track_api_call()
                
                if response.status == 200:
                    coins = await response.json()
                    
                    # First try to find exact symbol match
                    for coin in coins:
                        if coin['symbol'] == symbol:
                            _coin_id_cache[symbol] = coin['id']
                            return coin['id']
                            
                    # If no exact match, log warning
                    log.warning(f"No exact match found for symbol: {symbol}")
                    return None
                else:
                    log.error(f"Error fetching coin list: HTTP {response.status}")
                    return None
    except Exception as e:
        log.error(f"Error getting coin ID for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_coin_metrics(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch current market metrics for a coin."""
    try:
        coin_id = await get_coin_id(symbol)
        if not coin_id:
            log.error(f"Could not find CoinGecko ID for symbol: {symbol}")
            return None

        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE_URL}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'false',
                'sparkline': 'false'
            }
            
            headers = get_api_headers()
            async with session.get(url, params=params, headers=headers) as response:
                track_api_call()
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract market data
                    market_data = data.get('market_data', {})
                    metrics = {
                        'symbol': symbol,
                        'market_cap': market_data.get('market_cap', {}).get(DEFAULT_CURRENCY, 0),
                        'total_volume': market_data.get('total_volume', {}).get(DEFAULT_CURRENCY, 0),
                        'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                        'market_cap_rank': data.get('market_cap_rank', 0),
                        'community_score': data.get('community_data', {}).get('community_score', 0),
                        'public_interest_score': data.get('public_interest_stats', {}).get('alexa_rank', 0)
                    }
                    
                    # Store metrics in database
                    await async_bulk_insert(
                        data_list=[metrics], 
                        table_name='coingecko_metrics',
                        conflict_fields=['symbol'],
                        update_fields=['market_cap', 'total_volume', 'price_change_24h', 'market_cap_rank', 'community_score', 'public_interest_score']
                    )
                    log.info(f"Successfully stored metrics for {symbol}")
                    return metrics
                elif response.status == 429:
                    log.warning(f"Rate limit hit when fetching coin metrics for {symbol}")
                    await asyncio.sleep(5)  # Even with Pro plan, add small delay on rate limit
                    return None
                else:
                    log.error(f"Error fetching coin metrics: HTTP {response.status}")
                    return None
                    
    except Exception as e:
        log.error(f"Error fetching coin metrics for {symbol}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_coin_market_data(
    coin_id: str,
    vs_currency: str = DEFAULT_CURRENCY,
    days: str = "max",
    interval: str = "daily"
) -> Optional[Dict[str, List[float]]]:
    """Fetch historical market data for a coin."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': interval
            }
            
            headers = get_api_headers()
            async with session.get(url, params=params, headers=headers) as response:
                track_api_call()
                
                if response.status == 200:
                    data = await response.json()
                    
                    log.info(f"Successfully fetched market data for {coin_id} ({days} days, {interval} interval)")
                    
                    # Process and store historical data if needed
                    # Note: We're now returning the raw data without storing it to avoid duplicating storage
                    # The calling function backfill_coingecko_data will handle storage
                    return data
                
                elif response.status == 429:
                    log.warning(f"Rate limit hit when fetching market data for {coin_id}")
                    await asyncio.sleep(5)  # Even with Pro plan, add small delay on rate limit
                    return None
                else:
                    log.error(f"Error fetching market data: HTTP {response.status} for {coin_id}")
                    return None
                    
    except Exception as e:
        log.error(f"Error fetching market data for {coin_id}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_coin_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch current prices for a list of coin symbols from CoinGecko.
    
    Args:
        symbols: List of cryptocurrency symbols (e.g., 'BTC', 'ETH')
        
    Returns:
        Dictionary mapping symbols to their current USD prices
    """
    log.info(f"Fetching prices for {len(symbols)} symbols using CoinGecko Pro API")
    
    prices = {}
    rate_limiter = AsyncRateLimiter(API_CALLS_PER_MINUTE)
    
    # Process symbols in batches to optimize API calls
    # With Pro API we can use larger batches
    batch_size = 100  # Increased from 25 to 100 for Pro API
    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    for batch in symbol_batches:
        coin_ids = []
        symbol_to_id_map = {}
        
        # First map symbols to coin IDs
        for symbol in batch:
            try:
                await rate_limiter.wait_if_needed()
                coin_id = await get_coin_id(symbol.lower())
                
                if coin_id:
                    coin_ids.append(coin_id)
                    symbol_to_id_map[coin_id] = symbol
                else:
                    log.warning(f"Could not find CoinGecko ID for {symbol}, skipping price check")
            except Exception as e:
                log.error(f"Error mapping {symbol} to CoinGecko ID: {e}")
        
        if not coin_ids:
            continue
            
        # Get prices for all IDs in this batch in a single API call
        try:
            await rate_limiter.wait_if_needed()
            
            async with aiohttp.ClientSession() as session:
                ids_param = ",".join(coin_ids)
                url = f"{API_BASE_URL}/simple/price"
                params = {'ids': ids_param, 'vs_currencies': 'usd'}
                headers = get_api_headers()
                
                async with session.get(url, params=params, headers=headers) as response:
                    track_api_call()
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Map prices back to original symbols
                        for coin_id, price_data in data.items():
                            if 'usd' in price_data and coin_id in symbol_to_id_map:
                                symbol = symbol_to_id_map[coin_id]
                                prices[symbol] = price_data['usd']
                                log.debug(f"Price for {symbol}: ${price_data['usd']}")
                    elif response.status == 429:
                        log.warning("CoinGecko rate limit reached, waiting before retry")
                        await asyncio.sleep(5)  # Shorter wait with Pro plan
                    else:
                        log.error(f"Failed to get prices: HTTP {response.status}")
            
            # With Pro API, we can reduce the delay between batches
            await asyncio.sleep(0.5)  # Reduced from 2 seconds to 0.5 seconds
            
        except Exception as e:
            log.error(f"Error fetching prices for batch: {e}")
    
    # Log summary of prices found
    if prices:
        log.info(f"Successfully retrieved prices for {len(prices)} out of {len(symbols)} symbols")
        # Log all prices under $1
        cheap_coins = {s: p for s, p in prices.items() if p < 1.0}
        if cheap_coins:
            log.info(f"Found {len(cheap_coins)} coins under $1: {cheap_coins}")
    else:
        log.warning("Failed to retrieve any prices")
        
    return prices

async def get_trending_coins() -> List[Dict[str, Any]]:
    """
    Get trending coins on CoinGecko in the last 24 hours.
    
    Returns:
        List of trending coins with their details
    """
    log.info("Fetching trending coins from CoinGecko")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE_URL}/search/trending"
            headers = get_api_headers()
            
            async with session.get(url, headers=headers) as response:
                track_api_call()
                
                if response.status == 200:
                    data = await response.json()
                    coins = data.get('coins', [])
                    log.info(f"Successfully retrieved {len(coins)} trending coins")
                    
                    # Extract relevant data
                    trending_coins = []
                    for coin in coins:
                        coin_item = coin.get('item', {})
                        trending_coins.append({
                            'id': coin_item.get('id'),
                            'name': coin_item.get('name'),
                            'symbol': coin_item.get('symbol'),
                            'market_cap_rank': coin_item.get('market_cap_rank'),
                            'price_btc': coin_item.get('price_btc')
                        })
                    
                    return trending_coins
                else:
                    log.error(f"Error fetching trending coins: HTTP {response.status}")
                    return []
    except Exception as e:
        log.error(f"Error fetching trending coins: {e}")
        return []

async def get_global_market_data() -> Dict[str, Any]:
    """
    Get global cryptocurrency market data from CoinGecko.
    
    Returns:
        Dictionary with global market data
    """
    log.info("Fetching global market data from CoinGecko")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{API_BASE_URL}/global"
            headers = get_api_headers()
            
            async with session.get(url, headers=headers) as response:
                track_api_call()
                
                if response.status == 200:
                    data = await response.json()
                    global_data = data.get('data', {})
                    log.info("Successfully retrieved global market data")
                    
                    return global_data
                else:
                    log.error(f"Error fetching global market data: HTTP {response.status}")
                    return {}
    except Exception as e:
        log.error(f"Error fetching global market data: {e}")
        return {}