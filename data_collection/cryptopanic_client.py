"""
CryptoPanic API Client

This module handles interaction with the CryptoPanic API to fetch sentiment data.
"""

import os
import sys
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
import datetime
import pytz

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import config

log = logging.getLogger(__name__)

API_BASE_URL = "https://cryptopanic.com/api/v1"

async def fetch_market_sentiment(symbols: List[str]) -> Optional[Dict[str, Any]]:
    """
    Fetch market sentiment data from CryptoPanic API.
    
    Args:
        symbols: List of cryptocurrency symbols to fetch sentiment for
        
    Returns:
        Dictionary containing sentiment data if successful, None otherwise
    """
    if not config.CRYPTOPANIC_API_TOKEN:
        log.error("CryptoPanic API token not configured")
        return None
        
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                'auth_token': config.CRYPTOPANIC_API_TOKEN,
                'currencies': ','.join(symbols),
                'filter': 'important'
            }
            
            # Ensure the URL ends with trailing slash as required by CryptoPanic API
            url = f"{API_BASE_URL}/posts/"
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    log.error("CryptoPanic API authentication failed. Check your API token.")
                    return None
                else:
                    log.error(f"Error fetching CryptoPanic data: {response.status}")
                    return None
    except Exception as e:
        log.error(f"Exception in CryptoPanic API call: {e}")
        return None

async def fetch_news_sentiment(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch cryptocurrency news and sentiment data from CryptoPanic API.
    
    Args:
        symbol: The cryptocurrency symbol to fetch news for
        
    Returns:
        Dictionary containing news and sentiment data if successful, None otherwise
    """
    # Ensure the URL ends with trailing slash as required by CryptoPanic API
    url = f"{API_BASE_URL}/posts/"
    
    # Use currency parameter instead of currencies for single symbol
    params = {
        'auth_token': config.CRYPTOPANIC_API_TOKEN,
        'currencies': symbol,
        'filter': 'hot',
        'public': 'true',
        'kind': 'news'  # Specifically request news items
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Add current timestamp if not in response
                    if 'timestamp' not in data:
                        data['timestamp'] = datetime.datetime.now(pytz.utc).isoformat()
                    
                    return {
                        'symbol': symbol,
                        'timestamp': data.get('timestamp'),
                        'results': data.get('results', [])
                    }
                elif response.status == 401:
                    log.error("CryptoPanic API authentication failed. Check your API token.")
                    return None
                elif response.status == 404:
                    log.error(f"CryptoPanic API endpoint not found. URL: {url}")
                    # Log the full URL for debugging
                    full_url = url + "?" + "&".join([f"{k}={v}" for k, v in params.items() if k != 'auth_token'])
                    log.error(f"Full URL (without token): {full_url}")
                    return None
                else:
                    log.error(f"Error fetching CryptoPanic data: {response.status}")
                    return None
    except Exception as e:
        log.error(f"Exception in CryptoPanic fetch: {e}")
        return None

async def fetch_historical_ticker_news(tickers: List[str], date_str: str, items_per_page: int = 100, page: int = 1) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch historical news for specific tickers from CryptoPanic API.
    
    Args:
        tickers: List of cryptocurrency tickers to fetch news for
        date_str: Date string in format MMDDYYYY-MMDDYYYY
        items_per_page: Number of items per page
        page: Page number to fetch
        
    Returns:
        Tuple containing:
        - List of news items
        - Boolean indicating if there are more pages
    """
    url = f"{API_BASE_URL}/posts/"
    
    params = {
        'auth_token': config.CRYPTOPANIC_API_TOKEN,
        'currencies': ','.join(tickers),
        'public': 'true',
        'filter': 'hot',
        'date_range': date_str,
        'per_page': items_per_page,
        'page': page
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    
                    # Check if there are more pages
                    has_more_pages = data.get('next') is not None
                    
                    # Add source API and published_at fields
                    for item in results:
                        item['source_api'] = 'cryptopanic'
                        item['source_publisher'] = item.get('source', {}).get('title')
                        item['article_id'] = item.get('id')
                        item['text_content'] = item.get('title')  # Use title as text content
                        
                        # Convert published_at to datetime
                        if 'published_at' in item:
                            try:
                                published_at = datetime.datetime.fromisoformat(
                                    item['published_at'].replace('Z', '+00:00')
                                )
                                item['published_at'] = published_at
                            except (ValueError, TypeError):
                                # In case of parsing error, use current time
                                item['published_at'] = datetime.datetime.now(pytz.utc)
                    
                    return results, has_more_pages
                    
                else:
                    log.error(f"Error fetching historical news: {response.status}")
                    return [], False
    except Exception as e:
        log.error(f"Exception in CryptoPanic historical fetch: {e}")
        return [], False