#!/usr/bin/env python3
"""
Backfill API Data Script

This script fetches and stores historical data from various cryptocurrency APIs:
- CryptoPanic for sentiment data
- AlphaVantage for technical health metrics
- CoinGecko for market metrics

Usage:
    python backfill_api_data.py

Configuration (set in config.py):
    - BACKFILL_DAYS: Number of days of historical data to backfill
    - CRYPTOPANIC_CALLS_PER_MINUTE: API rate limit for CryptoPanic
    - ALPHAVANTAGE_CALLS_PER_MINUTE: API rate limit for AlphaVantage
    - COINGECKO_CALLS_PER_MINUTE: API rate limit for CoinGecko
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import config first to ensure it's available
import config

# Import custom rate limiter
from utils.rate_limiter import AsyncRateLimiter

# Then import other project modules
try:
    from data_collection.binance_client import get_target_symbols
    from data_collection.cryptopanic_client import fetch_market_sentiment, fetch_news_sentiment
    from data_collection.alphavantage_client import fetch_crypto_health_index
    from data_collection.coingecko_client import fetch_coin_metrics, fetch_coin_market_data
    from database.db_utils import store_cryptopanic_sentiment, store_alphavantage_health, store_coingecko_metrics
except ImportError as e:
    logging.error(f"Failed to import project modules. Check PYTHONPATH and module paths. Error: {e}")
    sys.exit(1)

# Basic logging config initially, might be overridden later in main
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

async def backfill_cryptopanic_data(symbols: List[str], start_date: datetime, end_date: datetime) -> None:
    """
    Backfills CryptoPanic sentiment data.
    
    Args:
        symbols: List of cryptocurrency symbols to fetch sentiment for
        start_date: Start date for the backfill period
        end_date: End date for the backfill period
    """
    log.info(f"Starting CryptoPanic backfill for symbols: {symbols}")
    limiter = AsyncRateLimiter(config.CRYPTOPANIC_CALLS_PER_MINUTE)
    
    # CryptoPanic doesn't support historical data via API, so we'll fetch current data
    # and store it with different timestamps to simulate historical data
    current_date = start_date
    while current_date <= end_date:
        log.debug(f"Processing CryptoPanic for timestamp: {current_date}")
        try:
            await limiter.wait_if_needed()
            
            # Process each symbol individually
            for symbol in symbols:
                sentiment_data = await fetch_news_sentiment(symbol)
                
                if sentiment_data and isinstance(sentiment_data, dict):
                    # Extract the required data
                    results = sentiment_data.get('results', [])
                    
                    # Calculate sentiment metrics
                    bullish_count = sum(1 for r in results if r.get('sentiment') == 'bullish')
                    bearish_count = sum(1 for r in results if r.get('sentiment') == 'bearish')
                    total_articles = len(results)
                    
                    # Calculate sentiment score (-1 to 1 range)
                    sentiment_score = 0
                    if total_articles > 0:
                        sentiment_score = (bullish_count - bearish_count) / total_articles
                    
                    # Store data with the current backfill date
                    data = {
                        'symbol': symbol,
                        'results': results,
                        'sentiment_score': sentiment_score,
                        'bullish_count': bullish_count,
                        'bearish_count': bearish_count,
                        'total_articles': total_articles,
                        'timestamp': current_date
                    }
                    
                    success = await store_cryptopanic_sentiment(data)
                    if success:
                        log.debug(f"Stored sentiment for {symbol} at {current_date}")
                    else:
                        log.warning(f"Failed to store sentiment for {symbol} at {current_date}")
                
                elif not sentiment_data:
                    log.warning(f"No sentiment data received from CryptoPanic for {symbol}")
                else:
                    log.error(f"Unexpected data type received from CryptoPanic: {type(sentiment_data)}")
                
                # Small delay between symbols
                await asyncio.sleep(0.5)
            
            # Increment date by one day
            current_date += timedelta(days=1)
            
        except Exception as e:
            log.error(f"Error backfilling CryptoPanic data at {current_date}: {e}", exc_info=True)
            await asyncio.sleep(10)  # Wait longer after an error
            # Continue to the next date instead of breaking
            current_date += timedelta(days=1)
    
    log.info("Finished CryptoPanic backfill.")

async def backfill_alphavantage_data(symbols: List[str], start_date: datetime, end_date: datetime) -> None:
    """
    Backfills AlphaVantage health metrics.
    
    Args:
        symbols: List of cryptocurrency symbols to fetch health metrics for
        start_date: Start date for the backfill period (not used by AlphaVantage API)
        end_date: End date for the backfill period (not used by AlphaVantage API)
    
    Note:
        AlphaVantage API does not support historical queries. This function will fetch
        current data and store it with the current timestamp.
    """
    log.info(f"Starting AlphaVantage backfill for symbols: {symbols}")
    # Note: AlphaVantage free tier has strict limits (e.g., 5 calls/min)
    limiter = AsyncRateLimiter(config.ALPHAVANTAGE_CALLS_PER_MINUTE)
    
    processed_count = 0
    for symbol in symbols:
        try:
            log.debug(f"Fetching AlphaVantage data for {symbol}")
            await limiter.wait_if_needed()
            
            # Fetch health index (current data only)
            health_index = await fetch_crypto_health_index(symbol)
            
            if health_index and isinstance(health_index, dict):
                # Prepare the data with proper defaults for missing values
                data = {
                    'symbol': symbol,
                    'health_score': float(health_index.get('health_score', 0)),
                    'rsi': float(health_index.get('rsi', 0)),
                    'macd': float(health_index.get('macd', 0)),
                    'macd_signal': float(health_index.get('macd_signal', 0)),
                    'timestamp': datetime.now(pytz.utc)
                }
                
                success = await store_alphavantage_health(data)
                if success:
                    log.info(f"Stored AlphaVantage health index for {symbol}")
                    processed_count += 1
                else:
                    log.warning(f"Failed to store AlphaVantage health index for {symbol}")
            elif health_index:
                log.warning(f"Received non-dict health index data for {symbol} from AlphaVantage: {health_index}")
            else:
                log.warning(f"No AlphaVantage health index data found for {symbol}")
                
            # Small delay between symbols
            await asyncio.sleep(0.5)
            
        except Exception as e:
            log.error(f"Error backfilling AlphaVantage data for {symbol}: {e}", exc_info=True)
            await asyncio.sleep(10)  # Wait longer after an error
    
    log.info(f"Finished AlphaVantage backfill. Stored {processed_count} records.")

async def backfill_coingecko_data(symbol_ids: List[str], start_date: datetime, end_date: datetime) -> None:
    """
    Backfills CoinGecko market data and metrics.
    
    Args:
        symbol_ids: List of CoinGecko symbol IDs (e.g., 'bitcoin', 'ethereum')
        start_date: Start date for the backfill period
        end_date: End date for the backfill period
    """
    log.info(f"Starting CoinGecko backfill for IDs: {symbol_ids}")
    limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)
    processed_metrics_count = 0
    processed_market_data_points = 0
    
    for symbol_id in symbol_ids:
        try:
            # --- Fetch historical market data ---
            log.debug(f"Fetching CoinGecko historical market data for {symbol_id}")
            await limiter.wait_if_needed()
            
            # Calculate days parameter for CoinGecko API (required as string)
            days_str = str((end_date - start_date).days + 1)
            
            # Call with proper parameter types (CoinGecko expects strings for parameters)
            market_data = await fetch_coin_market_data(
                coin_id=symbol_id, 
                vs_currency="usd",
                days=days_str, 
                interval="daily"
            )
            
            if market_data and isinstance(market_data, dict):
                log.info(f"Fetched historical market data for {symbol_id}")
                # TODO: Implement historical price storage if needed
                # This would typically involve iterating through the data points
                # and storing them in a dedicated price history table
                pass
            elif not market_data:
                log.warning(f"No historical market data found for {symbol_id}")
            else:
                log.error(f"Unexpected data type received for market data from CoinGecko: {type(market_data)}")
            
            # --- Fetch current metrics ---
            log.debug(f"Fetching CoinGecko current metrics for {symbol_id}")
            await limiter.wait_if_needed()
            metrics = await fetch_coin_metrics(symbol_id)
            
            if metrics and isinstance(metrics, dict):
                # Prepare data with proper defaults for missing values
                data = {
                    'symbol': symbol_id,
                    'market_cap': int(metrics.get('market_cap', 0)),
                    'total_volume': int(metrics.get('total_volume', 0)),
                    'price_change_24h': float(metrics.get('price_change_24h', 0)),
                    'market_cap_rank': int(metrics.get('market_cap_rank', 0)),
                    'community_score': float(metrics.get('community_score', 0)),
                    'public_interest_score': float(metrics.get('public_interest_score', 0)),
                    'timestamp': datetime.now(pytz.utc)
                }
                
                success = await store_coingecko_metrics(data)
                if success:
                    log.info(f"Stored current CoinGecko metrics for {symbol_id}")
                    processed_metrics_count += 1
                else:
                    log.warning(f"Failed to store CoinGecko metrics for {symbol_id}")
            elif metrics:
                log.warning(f"Received non-dict metrics data for {symbol_id} from CoinGecko: {metrics}")
            else:
                log.warning(f"No current CoinGecko metrics found for {symbol_id}")
            
            # Small delay between symbols
            await asyncio.sleep(0.5)
            
        except Exception as e:
            log.error(f"Error backfilling CoinGecko data for {symbol_id}: {e}", exc_info=True)
            await asyncio.sleep(10)  # Wait longer after an error
    
    log.info(f"Finished CoinGecko backfill. Stored {processed_metrics_count} metrics records.")

async def main_async():
    """Main asynchronous function to orchestrate the backfill process."""
    log.info("Starting main asynchronous backfill process...")

    # Get target symbols from Binance (e.g., ['BTCUSDT', 'ETHUSDT'])
    try:
        target_symbols_usdt = get_target_symbols()
        if not target_symbols_usdt:
            log.error("Failed to get target symbols from Binance (received empty list). Exiting.")
            return
        log.info(f"Retrieved {len(target_symbols_usdt)} target symbols from Binance.")
    except Exception as e:
        log.error(f"Failed to get target symbols from Binance: {e}", exc_info=True)
        return

    # Extract base symbols (e.g., 'btc', 'eth') - used by some APIs
    base_symbols_lower = list(set([s.replace('USDT', '').lower() for s in target_symbols_usdt]))
    # Symbols for APIs expecting uppercase (verify API docs)
    base_symbols_upper = [s.upper() for s in base_symbols_lower]

    # Map base symbols to CoinGecko IDs (e.g., 'btc' -> 'bitcoin')
    coingecko_ids = map_symbols_to_coingecko_ids(base_symbols_lower)
    if not coingecko_ids:
        log.warning("No symbols could be mapped to CoinGecko IDs. CoinGecko backfill will be skipped.")

    # Set date range for backfill
    end_date = datetime.now(pytz.utc)
    start_date = end_date - timedelta(days=config.BACKFILL_DAYS)
    log.info(f"Backfill date range: {start_date} to {end_date}")
    log.info(f"Target base symbols (uppercase): {base_symbols_upper}")
    log.info(f"Target CoinGecko IDs: {coingecko_ids}")

    # Create tasks for concurrent execution
    tasks = []
    if base_symbols_upper:
        tasks.append(backfill_cryptopanic_data(base_symbols_upper, start_date, end_date))
        tasks.append(backfill_alphavantage_data(base_symbols_upper, start_date, end_date))
    if coingecko_ids:
        tasks.append(backfill_coingecko_data(coingecko_ids, start_date, end_date))

    if not tasks:
        log.warning("No backfill tasks created. Exiting.")
        return

    # Run tasks concurrently
    log.info(f"Running {len(tasks)} backfill tasks concurrently...")
    await asyncio.gather(*tasks)

    log.info("All backfill tasks complete!")


def map_symbols_to_coingecko_ids(symbols: List[str]) -> List[str]:
    """
    Maps common crypto symbols (lowercase) to CoinGecko API IDs.
    **IMPORTANT:** This is a placeholder and needs a robust implementation.
    A simple dictionary is prone to errors and omissions.
    Consider fetching the /coins/list endpoint from CoinGecko periodically
    (https://docs.coingecko.com/reference/coins-list) to build and cache
    a reliable mapping between symbols and IDs.
    """
    log.info("Attempting to map symbols to CoinGecko IDs...")
    # Example static mapping - REPLACE WITH A DYNAMIC/ROBUST SOLUTION
    mapping = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'sol': 'solana',
        'bnb': 'binancecoin',
        'ada': 'cardano',
        'xrp': 'ripple',
        'doge': 'dogecoin',
        'dot': 'polkadot',
        'avax': 'avalanche-2',
        'matic': 'matic-network',
        # Add many more mappings...
    }
    mapped_ids = []
    unmapped_symbols = []
    for s in symbols:
        cg_id = mapping.get(s)
        if cg_id:
            mapped_ids.append(cg_id)
        else:
            unmapped_symbols.append(s)
            # Optionally, try a direct lookup via CoinGecko API if symbol often matches ID
            # Or just log the warning.

    if unmapped_symbols:
        log.warning(f"No CoinGecko ID mapping found for symbols: {unmapped_symbols}. They will be skipped for CoinGecko.")

    log.info(f"Successfully mapped {len(mapped_ids)} symbols to CoinGecko IDs.")
    return mapped_ids


if __name__ == "__main__":
    # Ensure essential config values are set
    required_configs = [
        'LOG_LEVEL', 'CRYPTOPANIC_CALLS_PER_MINUTE', 'ALPHAVANTAGE_CALLS_PER_MINUTE',
        'COINGECKO_CALLS_PER_MINUTE', 'BACKFILL_DAYS'
    ]
    missing_configs = [cfg for cfg in required_configs if not hasattr(config, cfg)]
    if missing_configs:
        # Log error even before full logging is configured
        print(f"ERROR: Configuration variables missing in config.py: {missing_configs}", file=sys.stderr)
        sys.exit(1)

    # Setup logging based on config level
    log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        force=True) # force=True to override initial basicConfig
    log = logging.getLogger(__name__) # Re-get logger after setting level

    # Validate numeric configs
    try:
        if config.CRYPTOPANIC_CALLS_PER_MINUTE <= 0 or \
           config.ALPHAVANTAGE_CALLS_PER_MINUTE <= 0 or \
           config.COINGECKO_CALLS_PER_MINUTE <= 0 or \
           config.BACKFILL_DAYS <= 0:
           raise ValueError("API calls per minute and backfill days must be positive.")
    except (AttributeError, ValueError, TypeError) as e:
        log.error(f"Invalid configuration value detected: {e}", exc_info=True)
        sys.exit(1)


    # Run the async main function using asyncio.run()
    try:
        log.info("Script starting...")
        # Check if running in a TTY environment for better KeyboardInterrupt handling
        if sys.stdin.isatty():
            log.info("Running in interactive mode. Press Ctrl+C to interrupt.")
        else:
            log.info("Running in non-interactive mode.")

        asyncio.run(main_async())

    except KeyboardInterrupt:
        log.info("Backfill process interrupted by user (Ctrl+C).")
        # Perform any necessary cleanup here
    except Exception as e:
        # Log the critical error with traceback
        log.critical(f"An unhandled error occurred during backfill: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero status code to indicate failure
    finally:
        log.info("Backfill script finished.")
        # Ensure logs are flushed if using file handlers, etc.