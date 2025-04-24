#!/usr/bin/env python3
"""
API Data Backfill Script - Now with async support

This script fetches and stores data from various cryptocurrency APIs
using async patterns and PostgreSQL optimizations.
Modified to accept symbol list from orchestration script.
"""

import os
import sys
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

import config
from utils.rate_limiter import AsyncRateLimiter
from database.db_utils import async_bulk_insert  # Changed from utils.database to database.db_utils
from data_collection.binance_client import get_target_symbols
from data_collection.cryptopanic_client import fetch_market_sentiment, fetch_news_sentiment, fetch_historical_ticker_news
from data_collection.coingecko_client import fetch_coin_metrics, fetch_coin_market_data, get_coin_id, fetch_coin_prices

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def validate_api_response(data: Dict[str, Any], data_type: str) -> bool:
    """
    Validate API response data based on its type.
    
    Args:
        data: The data to validate
        data_type: Type of data ('coingecko', 'cryptopanic')
        
    Returns:
        Boolean indicating whether the data is valid
    """
    if data is None:
        return False
        
    # Basic validation - ensure data is not empty
    if not data:
        log.warning(f"Empty {data_type} data received")
        return False
        
    # Specific validation based on data type
    if data_type == 'coingecko':
        # Check for required fields in CoinGecko data
        if 'prices' not in data:
            log.warning("Missing 'prices' in CoinGecko market data")
            return False
        if not isinstance(data['prices'], list) or len(data['prices']) == 0:
            log.warning("CoinGecko 'prices' data is empty or invalid format")
            return False
            
    elif data_type == 'cryptopanic':
        # Check for valid news items in CryptoPanic data
        if not isinstance(data, list):
            log.warning("CryptoPanic data should be a list of news items")
            return False
        if len(data) == 0:
            log.warning("No news items in CryptoPanic data")
            return False
            
    return True

async def backfill_cryptopanic_data(symbols: List[str], start_date: datetime, end_date: datetime) -> None:
    """Backfill sentiment data from CryptoPanic API, calculating a score."""
    log.info(f"Starting CryptoPanic backfill for {len(symbols)} symbols")
    
    rate_limiter = AsyncRateLimiter(config.CRYPTOPANIC_CALLS_PER_MINUTE)
    
    # Format date range in MM/DD/YYYY-MM/DD/YYYY format as required by CryptoPanic API
    date_range = f"{start_date.strftime('%m/%d/%Y')}-{end_date.strftime('%m/%d/%Y')}"
    log.info(f"Using date range: {date_range}")
    
    try:
        for symbol in symbols:
            # Process each symbol with proper rate limiting
            await rate_limiter.wait_if_needed()
            
            # Fetch historical news sentiment for this symbol
            page = 1
            has_more_pages = True
            
            while has_more_pages:
                log.info(f"Fetching page {page} for {symbol} in date range {date_range}")
                results, has_more_pages = await fetch_historical_ticker_news(
                    tickers=[symbol],
                    date_str=date_range,
                    items_per_page=100,
                    page=page
                )
                
                if not validate_api_response(results, 'cryptopanic'):
                    log.warning(f"Invalid CryptoPanic data for {symbol} (page {page})")
                    break
                
                if results:
                    log.info(f"Got {len(results)} news items for {symbol} (page {page})")
                    
                    # Prepare data for database, calculating sentiment score
                    data_list = []
                    bullish_count = 0
                    bearish_count = 0
                    total_articles = len(results)

                    for item in results:
                        sentiment_label = item.get('votes', {}).get('sentiment', 'neutral').lower()
                        if sentiment_label == 'bullish':
                            bullish_count += 1
                        elif sentiment_label == 'bearish':
                            bearish_count += 1
                        
                        # Convert published_at to datetime
                        pub_date_str = item.get('published_at')
                        pub_date = None
                        if pub_date_str:
                            try:
                                pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            except ValueError:
                                log.warning(f"Could not parse date '{pub_date_str}' for CryptoPanic item {item.get('id')}")
                                continue # Skip if date is invalid
                        else:
                            log.warning(f"CryptoPanic item {item.get('id')} missing published_at date, skipping.")
                            continue

                        record = {
                            'symbol': symbol,
                            'source': 'cryptopanic',
                            'content_id': item['id'],
                            'title': item.get('title', ''),
                            'url': item.get('url', ''),
                            'published_at': pub_date,
                            'raw_sentiment_label': sentiment_label, # Store the raw label if needed
                            'processed_at': datetime.now(pytz.UTC)
                        }
                        data_list.append(record)
                    
                    # Calculate overall sentiment score for this batch/page
                    sentiment_score = 0
                    if total_articles > 0:
                        sentiment_score = (bullish_count - bearish_count) / total_articles

                    # Add score to records (or store separately if schema requires)
                    # Assuming 'api_data' table can store this score. If not, schema needs update.
                    # For simplicity, let's assume we store the score with each record for this batch.
                    # A better approach might be a separate aggregate table.
                    for r in data_list:
                        r['calculated_sentiment_score'] = sentiment_score

                    if data_list:
                        await async_bulk_insert(
                            data_list=data_list,
                            table_name='api_data', # Ensure this table has 'raw_sentiment_label' and 'calculated_sentiment_score' columns
                            conflict_fields=['source', 'content_id'],
                            update_fields=['raw_sentiment_label', 'calculated_sentiment_score', 'processed_at']
                        )
                        log.info(f"Stored {len(data_list)} CryptoPanic items for {symbol} (page {page}) with score {sentiment_score:.2f}")
                else:
                    log.warning(f"No news data for {symbol} in date range (page {page})")
                    # Break the loop if no results were found
                    break
                
                # Increment page for next iteration
                page += 1
                
                # Add small delay between API calls
                await asyncio.sleep(1)
            
            # Add delay between symbols
            await asyncio.sleep(2)
            
    except Exception as e:
        log.error(f"Error backfilling CryptoPanic data: {e}", exc_info=True)
        await asyncio.sleep(10)

    log.info("Finished CryptoPanic backfill.")

async def backfill_coingecko_data(symbols: List[str], start_date: datetime, end_date: datetime) -> None:
    """Backfill market metrics from CoinGecko API."""
    log.info(f"Starting CoinGecko backfill for {len(symbols)} symbols")
    
    # Create a rate limiter specifically for this function
    rate_limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)
    
    for symbol in symbols:
        try:
            # Wait for rate limiting
            await rate_limiter.wait_if_needed()
            
            # Skip if the symbol doesn't match expected pattern (should end with USDT)
            if not symbol.endswith('USDT'):
                log.warning(f"Symbol {symbol} doesn't follow the expected pattern (should end with USDT), skipping")
                continue
                
            # Convert trading pair to base symbol (e.g., BTCUSDT -> BTC)
            base_symbol = symbol.replace('USDT', '').lower()
            
            # Validate base symbol is not empty
            if not base_symbol:
                log.warning(f"Invalid symbol format: {symbol}, skipping")
                continue
                
            log.info(f"Processing {symbol} -> base symbol: {base_symbol}")
            
            # Get proper coin ID from CoinGecko
            coin_id = await get_coin_id(base_symbol)
            if not coin_id:
                log.warning(f"Could not find CoinGecko ID for {symbol} (base: {base_symbol}), skipping")
                continue
                
            log.info(f"Fetching historical market data for {symbol} (CoinGecko ID: {coin_id})")
            
            # Calculate days parameter based on date range
            days_diff = (end_date - start_date).days + 1
            days_param = str(days_diff) if days_diff <= 365 else "max"
            
            market_data = await fetch_coin_market_data(
                coin_id=coin_id,
                days=days_param,
                interval="daily"
            )
            
            if not validate_api_response(market_data, 'coingecko'):
                log.warning(f"Invalid CoinGecko data for {symbol}")
                continue
            
            if market_data and 'prices' in market_data:
                price_count = len(market_data.get('prices', []))
                log.info(f"Retrieved {price_count} price data points for {symbol}")
                
                # Convert data to bytes for database storage
                import json
                market_data_str = json.dumps(market_data)
                market_data_bytes = market_data_str.encode('utf-8')
                
                # Store in database
                record = {
                    'symbol': symbol,
                    'source': 'coingecko',
                    'data': market_data_bytes,
                    'processed_at': datetime.now(pytz.UTC)
                }
                
                # Using the updated function signature
                await async_bulk_insert(
                    data_list=[record],
                    table_name='api_data',
                    conflict_fields=['symbol', 'source'],
                    update_fields=['data', 'processed_at']
                )
            else:
                log.warning(f"No market data retrieved for {symbol}")
            
            # CoinGecko free tier is limited to 10-50 calls per minute
            # Add a delay between coin requests to avoid rate limits
            await asyncio.sleep(6)  # Increased delay to avoid rate limits
            
        except Exception as e:
            log.error(f"Error backfilling CoinGecko data for {symbol}: {e}", exc_info=True)
            # On error, add a longer delay to avoid potential rate limit issues
            await asyncio.sleep(30)

    log.info("Finished CoinGecko backfill.")

async def main(symbols: Optional[List[str]] = None):
    """
    Main function to run API data backfill.
    
    Args:
        symbols: Optional list of symbols to backfill. If not provided,
                 will use low-value coins (< $1) from CoinGecko.
                 If provided, will still filter them to ensure only coins under $1 are processed.
    """
    log.info("Starting main asynchronous backfill process...")

    # Define API keys dictionary for multiple services
    api_keys = {
        'CryptoPanic': config.CRYPTOPANIC_API_TOKEN or None,  # Using the CRYPTOPANIC_API_TOKEN instead of CRYPTOPANIC_API_KEY
        'CoinGecko': config.COINGECKO_API_KEY or None
    }
    
    missing_credentials = [api for api, key in api_keys.items() if not key]
    if missing_credentials:
        log.warning(f"Missing API credentials for: {', '.join(missing_credentials)}")
    
    if not symbols:
        # If no symbols provided, fetch low-value coins from CoinGecko
        log.info("No symbols provided, fetching low-value coins from CoinGecko")
        try:
            coins = await fetch_coin_prices(symbols=[])
            if not coins:
                log.error("Failed to fetch coin list from CoinGecko")
                return
            symbols = [coin.upper() for coin, price in coins.items() if price < 1.0]
            log.info(f"Found {len(symbols)} coins with price < $1")
        except Exception as e:
            log.error(f"Error fetching coin list: {e}")
            return
    else:
        log.info(f"Filtering {len(symbols)} provided symbols to only include coins under $1")
        # Convert symbols to base symbols (remove USDT suffix)
        base_symbols = [s.replace('USDT', '') for s in symbols]
        
        try:
            # Fetch prices for the provided symbols
            coins = await fetch_coin_prices(symbols=base_symbols)
            if not coins:
                log.error("Failed to fetch coin prices from CoinGecko")
                return
            
            # Filter to only keep symbols with price < $1
            filtered_symbols = []
            for symbol in symbols:
                base = symbol.replace('USDT', '')
                if base in coins and coins[base] < 1.0:
                    filtered_symbols.append(symbol)
            
            # Update symbols list with only low-value coins
            symbols = filtered_symbols
            log.info(f"After filtering, found {len(symbols)} coins with price < $1")
            
            if not symbols:
                log.warning("No coins under $1 found in the provided symbol list, exiting backfill")
                return
        except Exception as e:
            log.error(f"Error filtering low-value coins: {e}")
            return
    
    # Set date range for backfill
    end_date = datetime.now(pytz.utc)
    start_date = end_date - timedelta(days=config.BACKFILL_DAYS)
    log.info(f"Setting backfill date range: {start_date.date()} to {end_date.date()}")

    # Run backfills concurrently for each API
    try:
        await asyncio.gather(
            backfill_cryptopanic_data(symbols, start_date, end_date) if hasattr(config, 'CRYPTOPANIC_API_KEY') and config.CRYPTOPANIC_API_KEY else asyncio.sleep(0),
            backfill_coingecko_data(symbols, start_date, end_date)
        )
        log.info("API data backfill completed successfully")
    except Exception as e:
        log.error(f"Error during API data backfill: {e}")
        raise

if __name__ == "__main__":
    try:
        # Validate numeric configs
        if not all([
            config.CRYPTOPANIC_CALLS_PER_MINUTE > 0,
            config.COINGECKO_CALLS_PER_MINUTE > 0,
            config.BACKFILL_DAYS > 0
        ]):
            raise ValueError("API calls per minute and backfill days must be positive.")

        # Run the async main function
        asyncio.run(main())

    except KeyboardInterrupt:
        log.info("Backfill process interrupted by user.")
    except Exception as e:
        log.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("Backfill script finished.")