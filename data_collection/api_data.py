# /data_collection/api_data.py

"""
API Data module for fetching cryptocurrency market data from external APIs.

This module provides functions to fetch market sentiment, health indices,
and various crypto metrics from different data providers.
"""

import logging
import datetime
import pytz
from typing import Dict, Any, List, Optional
import requests
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential
import json

# Use absolute import to avoid circular imports
import config
from database.db_utils import store_cryptopanic_sentiment, store_alphavantage_health, store_coingecko_metrics
from utils.rate_limiter import AsyncRateLimiter

log = logging.getLogger(__name__)

# Initialize rate limiters based on API limits
cryptopanic_limiter = AsyncRateLimiter(config.CRYPTOPANIC_CALLS_PER_MINUTE)
alphavantage_limiter = AsyncRateLimiter(config.ALPHAVANTAGE_CALLS_PER_MINUTE)
coingecko_limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_market_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """
    Fetches market sentiment data from CryptoPanic API for given symbols.
    
    Args:
        symbols: List of cryptocurrency symbols (e.g., ['BTC', 'ETH'])
        
    Returns:
        Dict containing sentiment scores and other metrics
    """
    if not symbols:
        log.warning("No symbols provided for sentiment analysis")
        return {}
        
    # Use the first symbol for demonstration
    symbol = symbols[0]
    
    try:
        # Properly wait for rate limiter before making the request
        await cryptopanic_limiter.wait_if_needed()
        
        log.debug(f"Fetching CryptoPanic sentiment data for {symbol}")
        
        # API endpoint for CryptoPanic
        url = f"https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': config.CRYPTOPANIC_API_TOKEN,
            'currencies': symbol,
            'public': 'true',
            'kind': 'news'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Process the results
        results = data.get('results', [])
        bullish_count = sum(1 for r in results if r.get('sentiment') == 'bullish')
        bearish_count = sum(1 for r in results if r.get('sentiment') == 'bearish')
        total_articles = len(results)
        
        # Calculate sentiment score (-1 to 1 range)
        sentiment_score = 0
        if total_articles > 0:
            sentiment_score = (bullish_count - bearish_count) / total_articles
            
        sentiment_data = {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'total_articles': total_articles,
            'results': results,  # Include the actual results
            'timestamp': datetime.datetime.now(pytz.utc)
        }
        
        # Store the sentiment data asynchronously
        try:
            await store_cryptopanic_sentiment(sentiment_data)
        except Exception as e:
            log.error(f"Failed to store CryptoPanic sentiment data: {e}")
            
        return sentiment_data
        
    except RequestException as e:
        log.error(f"Error fetching CryptoPanic data: {e}")
        return {}
    except Exception as e:
        log.error(f"Unexpected error processing CryptoPanic data: {e}", exc_info=True)
        return {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_crypto_health_index(symbol: str) -> Dict[str, Any]:
    """
    Fetches crypto health index data from AlphaVantage API for a given symbol.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC')
        
    Returns:
        Dict containing health score and other technical indicators
    """
    try:
        # Wait for rate limiter before making the request
        await alphavantage_limiter.wait_if_needed()
        
        log.debug(f"Fetching AlphaVantage health data for {symbol}")
        
        # API endpoint for AlphaVantage
        url = config.ALPHAVANTAGE_BASE_URL
        params = {
            'function': 'CRYPTO_RATING',
            'symbol': symbol,
            'apikey': config.ALPHAVANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract health metrics
        crypto_rating = data.get('Crypto Rating (FCAS)', {})
        health_score = float(crypto_rating.get('FCAS', 0))
        
        # Wait again before fetching RSI
        await alphavantage_limiter.wait_if_needed()
        
        # Also get RSI and MACD for the symbol
        params = {
            'function': 'RSI',
            'symbol': f"CRYPTO:{symbol}",
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': config.ALPHAVANTAGE_API_KEY
        }
        
        rsi_response = requests.get(url, params=params, timeout=10)
        rsi_response.raise_for_status()
        rsi_data = rsi_response.json()
        
        # Extract RSI value from the most recent data point
        rsi_values = rsi_data.get('Technical Analysis: RSI', {})
        latest_date = sorted(rsi_values.keys())[-1] if rsi_values else None
        rsi = float(rsi_values[latest_date]['RSI']) if latest_date else 50
        
        # Wait again before fetching MACD
        await alphavantage_limiter.wait_if_needed()
        
        # Get MACD data
        params = {
            'function': 'MACD',
            'symbol': f"CRYPTO:{symbol}",
            'interval': 'daily',
            'series_type': 'close',
            'apikey': config.ALPHAVANTAGE_API_KEY
        }
        
        macd_response = requests.get(url, params=params, timeout=10)
        macd_response.raise_for_status()
        macd_data = macd_response.json()
        
        # Extract MACD values from the most recent data point
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
        
        # Store the health data asynchronously
        try:
            await store_alphavantage_health(health_data)
        except Exception as e:
            log.error(f"Failed to store AlphaVantage health data: {e}")
            
        return health_data
        
    except RequestException as e:
        log.error(f"Error fetching AlphaVantage data: {e}")
        return {}
    except Exception as e:
        log.error(f"Unexpected error processing AlphaVantage data: {e}", exc_info=True)
        return {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_coin_metrics(symbol: str) -> Dict[str, Any]:
    """
    Fetches coin metrics data from CoinGecko API for a given symbol.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC')
        
    Returns:
        Dict containing market cap, volume, price changes, and community metrics
    """
    try:
        # Wait for rate limiter before making the request
        await coingecko_limiter.wait_if_needed()
        
        log.debug(f"Fetching CoinGecko metrics for {symbol}")
        
        # Convert common symbols to CoinGecko IDs
        symbol_to_id = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot'
        }
        
        coin_id = symbol_to_id.get(symbol, symbol.lower())
        
        # API endpoint for CoinGecko
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'false'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract market data
        market_data = data.get('market_data', {})
        market_cap = market_data.get('market_cap', {}).get('usd', 0)
        total_volume = market_data.get('total_volume', {}).get('usd', 0)
        price_change_24h = market_data.get('price_change_percentage_24h', 0)
        price_change_7d = market_data.get('price_change_percentage_7d', 0)
        market_cap_rank = data.get('market_cap_rank', 0)
        
        # Extract community data
        community_data = data.get('community_data', {})
        community_score = community_data.get('community_score', 0)
        
        # Extract public interest stats
        public_interest_stats = data.get('public_interest_stats', {})
        public_interest_score = public_interest_stats.get('alexa_rank', 0)
        
        # Invert alexa rank to get a score where higher is better
        if public_interest_score > 0:
            public_interest_score = 1000000 / public_interest_score
        
        metrics_data = {
            'symbol': symbol,
            'market_cap': market_cap,
            'total_volume': total_volume,
            'price_change_24h': price_change_24h,
            'price_change_7d': price_change_7d,
            'market_cap_rank': market_cap_rank,
            'community_score': community_score,
            'public_interest_score': public_interest_score,
            'timestamp': datetime.datetime.now(pytz.utc)
        }
        
        # Store the metrics data asynchronously
        try:
            await store_coingecko_metrics(metrics_data)
        except Exception as e:
            log.error(f"Failed to store CoinGecko metrics data: {e}")
            
        return metrics_data
        
    except RequestException as e:
        log.error(f"Error fetching CoinGecko data: {e}")
        return {}
    except Exception as e:
        log.error(f"Unexpected error processing CoinGecko data: {e}", exc_info=True)
        return {}