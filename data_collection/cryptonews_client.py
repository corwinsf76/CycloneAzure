# /data_collection/cryptonews_client.py

import logging
import datetime
import pytz
from typing import List, Dict, Optional, Tuple, Any
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import json

import config
from database.db_utils import async_bulk_insert

log = logging.getLogger(__name__)

# --- CryptoNews API Configuration ---
CRYPTONEWS_API_TOKEN = config.CRYPTONEWS_API_TOKEN
CRYPTONEWS_BASE_URL = "https://cryptonews-api.com/api/v1"

# Define the source timezone (Eastern Time) - Using pytz for DST handling
SOURCE_TIMEZONE_STR = 'US/Eastern'
try:
    SOURCE_TIMEZONE = pytz.timezone(SOURCE_TIMEZONE_STR)
except pytz.exceptions.UnknownTimeZoneError:
    log.error(f"Could not find timezone {SOURCE_TIMEZONE_STR}. Falling back to UTC.")
    SOURCE_TIMEZONE = pytz.UTC

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _make_api_request(endpoint_path: str, params: Dict) -> Optional[Dict]:
    """Helper function to make requests and handle common errors with retry logic."""
    if not CRYPTONEWS_API_TOKEN:
        log.error("CryptoNews API Token is not configured.")
        return None

    params['token'] = CRYPTONEWS_API_TOKEN
    url = f"{CRYPTONEWS_BASE_URL}{endpoint_path}"

    log.debug(f"Making CryptoNews API request to {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as response:
                log.debug(f"CryptoNews API response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    if 'error' in data and data['error']:
                        log.error(f"CryptoNews API returned an error: {data['error']}")
                        return None
                    return data
                else:
                    log.error(f"CryptoNews API request failed with status {response.status}")
                    return None
                    
    except asyncio.TimeoutError:
        log.error(f"Timeout error fetching data from CryptoNews API: {url}")
    except Exception as e:
        log.error(f"Error processing CryptoNews request: {e}", exc_info=True)
    
    return None

async def fetch_ticker_news(tickers: List[str], items_per_page: int = 50) -> List[Dict]:
    """Fetch latest news for specified tickers."""
    params = {
        'tickers': ','.join(tickers),
        'items': items_per_page,
        'page': 1,
        'type': 'article'
    }

    data = await _make_api_request('/news', params)
    if not data:
        return []

    news_items = []
    for item in data.get('data', []):
        # Convert naive datetime to timezone-aware
        published_dt = datetime.datetime.strptime(
            item['date'],
            '%Y-%m-%d %H:%M:%S'
        ).replace(tzinfo=SOURCE_TIMEZONE).astimezone(pytz.UTC)

        processed_item = {
            'source_api': 'cryptonews',
            'source_publisher': item.get('source_name'),
            'article_id': str(item.get('news_id')),
            'title': item.get('title'),
            'text_content': item.get('text'),
            'url': item.get('news_url'),
            'published_at': published_dt,
            'tickers_mentioned': item.get('tickers', []),
            'fetched_at': datetime.datetime.now(pytz.utc)
        }
        news_items.append(processed_item)

    # Store news items in database
    if news_items:
        try:
            await async_bulk_insert(news_items, 'news_data')
            log.info(f"Successfully stored {len(news_items)} news items")
        except Exception as e:
            log.error(f"Error storing news items in database: {e}")

    return news_items

async def fetch_historical_ticker_news(
    tickers: List[str],
    date_str: str,
    items_per_page: int = 100,
    page: int = 1
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch historical news for specified tickers.
    Returns tuple of (news_items, has_more_pages).
    """
    params = {
        'tickers': ','.join(tickers),
        'items': items_per_page,
        'page': page,
        'date': date_str,
        'type': 'article'
    }

    data = await _make_api_request('/news', params)
    if not data:
        return [], False

    news_items = []
    for item in data.get('data', []):
        # Convert naive datetime to timezone-aware
        published_dt = datetime.datetime.strptime(
            item['date'],
            '%Y-%m-%d %H:%M:%S'
        ).replace(tzinfo=SOURCE_TIMEZONE).astimezone(pytz.UTC)

        processed_item = {
            'source_api': 'cryptonews',
            'source_publisher': item.get('source_name'),
            'article_id': str(item.get('news_id')),
            'title': item.get('title'),
            'text_content': item.get('text'),
            'url': item.get('news_url'),
            'published_at': published_dt,
            'tickers_mentioned': item.get('tickers', []),
            'fetched_at': datetime.datetime.now(pytz.utc)
        }
        news_items.append(processed_item)

    # Determine if there are more pages
    total_pages = data.get('total_pages', 1)
    has_more_pages = page < total_pages

    # Store news items in database
    if news_items:
        try:
            await async_bulk_insert(news_items, 'news_data')
            log.info(f"Successfully stored {len(news_items)} historical news items from page {page}")
        except Exception as e:
            log.error(f"Error storing historical news items in database: {e}")

    return news_items, has_more_pages

async def fetch_crypto_news(
    symbol: str,
    from_date: Optional[datetime.datetime] = None,
    to_date: Optional[datetime.datetime] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Fetch crypto news for a specific symbol/ticker for the dashboard.
    
    Args:
        symbol: Ticker symbol to fetch news for (e.g., 'BTC')
        from_date: Start date for news articles
        to_date: End date for news articles
        limit: Maximum number of articles to fetch
        
    Returns:
        List of news articles with metadata
    """
    try:
        log.info(f"Fetching crypto news for {symbol} from {from_date} to {to_date}")
        
        # Format the symbol for API request
        # Strip USDT if present (convert BTCUSDT to BTC)
        if symbol.endswith('USDT'):
            ticker = symbol[:-4]
        else:
            ticker = symbol
            
        # If from_date is provided, fetch historical news
        if from_date:
            # Format date for API request (YYYY-MM-DD)
            date_str = from_date.strftime('%Y-%m-%d')
            news_items, _ = await fetch_historical_ticker_news(
                tickers=[ticker],
                date_str=date_str,
                items_per_page=limit,
                page=1
            )
        else:
            # Fetch latest news
            news_items = await fetch_ticker_news(
                tickers=[ticker],
                items_per_page=limit
            )
            
        return news_items
        
    except Exception as e:
        log.error(f"Error fetching crypto news for {symbol}: {e}")
        return []