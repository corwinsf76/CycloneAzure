# /data_collection/api_data.py

"""
API Data module for fetching cryptocurrency market data from external APIs.
Now optimized for PostgreSQL and async operations.
"""

import logging
import datetime
import pytz
from typing import Dict, Any, List, Optional
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import asyncio

import config
from database.db_utils import async_bulk_insert
from utils.rate_limiter import AsyncRateLimiter

log = logging.getLogger(__name__)

# Initialize rate limiters based on API limits
cryptopanic_limiter = AsyncRateLimiter(config.CRYPTOPANIC_CALLS_PER_MINUTE)
alphavantage_limiter = AsyncRateLimiter(config.ALPHAVANTAGE_CALLS_PER_MINUTE)
coingecko_limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_market_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """Fetch market sentiment data from CryptoPanic API."""
    try:
        await cryptopanic_limiter.wait_if_needed()
        
        async with aiohttp.ClientSession() as session:
            url = "https://cryptopanic.com/api/v1/posts"
            params = {
                'auth_token': config.CRYPTOPANIC_API_KEY,
                'currencies': ','.join(symbols),
                'public': 'true',
                'kind': 'news'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    bullish_count = sum(1 for r in results if r.get('sentiment') == 'bullish')
                    bearish_count = sum(1 for r in results if r.get('sentiment') == 'bearish')
                    total_articles = len(results)
                    
                    # Calculate sentiment score (-1 to 1 range)
                    sentiment_score = 0
                    if total_articles > 0:
                        sentiment_score = (bullish_count - bearish_count) / total_articles
                        
                    sentiment_data = {
                        'symbol': ','.join(symbols),
                        'sentiment_score': sentiment_score,
                        'bullish_count': bullish_count,
                        'bearish_count': bearish_count,
                        'total_articles': total_articles,
                        'timestamp': datetime.datetime.now(pytz.utc)
                    }
                    
                    # Store using async bulk insert
                    await async_bulk_insert([sentiment_data], 'cryptopanic_sentiment')
                    return sentiment_data
                else:
                    log.error(f"Error fetching CryptoPanic data: {response.status}")
                    return {}
    except Exception as e:
        log.error(f"Unexpected error processing CryptoPanic data: {e}", exc_info=True)
        return {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_crypto_health_index(symbol: str) -> Dict[str, Any]:
    """Fetch health index data from AlphaVantage API."""
    try:
        await alphavantage_limiter.wait_if_needed()
        
        async with aiohttp.ClientSession() as session:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'CRYPTO_RATING',
                'symbol': symbol,
                'apikey': config.ALPHAVANTAGE_API_KEY
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    crypto_rating = data.get('Crypto Rating (FCAS)', {})
                    health_score = float(crypto_rating.get('FCAS', 0))
                    
                    # Get RSI
                    await alphavantage_limiter.wait_if_needed()
                    params['function'] = 'RSI'
                    params['interval'] = 'daily'
                    params['time_period'] = 14
                    params['series_type'] = 'close'
                    
                    async with session.get(url, params=params) as rsi_response:
                        rsi_data = await rsi_response.json()
                        rsi_values = rsi_data.get('Technical Analysis: RSI', {})
                        latest_date = sorted(rsi_values.keys())[-1] if rsi_values else None
                        rsi = float(rsi_values[latest_date]['RSI']) if latest_date else 50
                    
                    # Get MACD
                    await alphavantage_limiter.wait_if_needed()
                    params['function'] = 'MACD'
                    
                    async with session.get(url, params=params) as macd_response:
                        macd_data = await macd_response.json()
                        macd_values = macd_data.get('Technical Analysis: MACD', {})
                        latest_date = sorted(macd_values.keys())[-1] if macd_values else None
                        macd = float(macd_values[latest_date]['MACD']) if latest_date else 0
                        macd_signal = float(macd_values[latest_date]['MACD_Signal']) if latest_date else 0
                    
                    health_data = {
                        'symbol': symbol,
                        'health_score': health_score,
                        'rsi': rsi,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'timestamp': datetime.datetime.now(pytz.utc)
                    }
                    
                    # Store using async bulk insert
                    await async_bulk_insert([health_data], 'alphavantage_health')
                    return health_data
                else:
                    log.error(f"Error fetching AlphaVantage data: {response.status}")
                    return {}
    except Exception as e:
        log.error(f"Unexpected error processing AlphaVantage data: {e}", exc_info=True)
        return {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_coin_metrics(symbol: str) -> Dict[str, Any]:
    """Fetch market metrics data from CoinGecko API."""
    try:
        await coingecko_limiter.wait_if_needed()
        
        async with aiohttp.ClientSession() as session:
            # Convert symbol to CoinGecko ID (e.g., 'btc' -> 'bitcoin')
            coin_id = symbol.lower()  # You might need a more sophisticated mapping
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'false'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract market data
                    market_data = data.get('market_data', {})
                    metrics_data = {
                        'symbol': symbol,
                        'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                        'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                        'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                        'market_cap_rank': data.get('market_cap_rank', 0),
                        'community_score': data.get('community_data', {}).get('community_score', 0),
                        'public_interest_score': _calculate_interest_score(
                            data.get('public_interest_stats', {}).get('alexa_rank', 0)
                        ),
                        'timestamp': datetime.datetime.now(pytz.utc)
                    }
                    
                    # Store using async bulk insert
                    await async_bulk_insert([metrics_data], 'coingecko_metrics')
                    return metrics_data
                else:
                    log.error(f"Error fetching CoinGecko data: {response.status}")
                    return {}
    except Exception as e:
        log.error(f"Unexpected error processing CoinGecko data: {e}", exc_info=True)
        return {}

def _calculate_interest_score(alexa_rank: int) -> float:
    """Calculate public interest score from Alexa rank."""
    if alexa_rank > 0:
        return 1000000 / alexa_rank
    return 0