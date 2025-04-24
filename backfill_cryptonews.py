#!/usr/bin/env python3
"""
CryptoNews Data Backfill Script - Now with async support

This script fetches and stores historical crypto news data
using async patterns and PostgreSQL optimizations.
Uses config.BACKFILL_DAYS for time range consistency with other scripts.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any, Optional
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import config
from utils.rate_limiter import AsyncRateLimiter
from data_collection.cryptonews_client import fetch_ticker_news, fetch_historical_ticker_news
from database.db_utils import async_bulk_insert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def backfill_crypto_news(symbols: Optional[List[str]] = None, start_date: Optional[datetime] = None) -> None:
    """
    Backfill crypto news data and perform sentiment analysis.
    
    Args:
        symbols: List of cryptocurrency symbols to fetch news for. If None, uses config.BASE_SYMBOLS.
        start_date: Start date for backfill period. If None, uses 7 days before current date.
    """
    if symbols is None:
        symbols = config.BASE_SYMBOLS
        log.info(f"Using {len(symbols)} symbols from config")
    else:
        log.info(f"Using {len(symbols)} provided symbols")
    
    # Filter symbols to only include coins under $1
    log.info("Filtering symbols to only include coins under $1")
    from data_collection.coingecko_client import fetch_coin_prices
    
    try:
        # Convert symbols to base symbols (remove USDT suffix)
        base_symbols = [s.replace('USDT', '') for s in symbols]
        
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
    
    if start_date is None:
        # Default to 7 days ago if not specified
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=config.BACKFILL_DAYS)
        log.info(f"Using default backfill period of {config.BACKFILL_DAYS} days")
    else:
        end_date = datetime.now(pytz.UTC)
        log.info(f"Using custom backfill period from {start_date.date()} to {end_date.date()}")
    
    # Format dates for API - MM/DD/YYYY-MM/DD/YYYY format for CryptoNews API
    date_format = "%m/%d/%Y"
    date_range = f"{start_date.strftime(date_format)}-{end_date.strftime(date_format)}"
    
    log.info(f"Starting crypto news backfill for {len(symbols)} symbols with date range: {date_range}")
    
    # Process symbols in batches to avoid overwhelming the API
    batch_size = 5
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        log.info(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {batch}")
        
        for symbol in batch:
            try:
                log.info(f"Fetching news for {symbol} with date range: {date_range}")
                
                # Use fetch_historical_ticker_news instead of fetch_crypto_news
                page = 1
                has_more_pages = True
                total_items = 0
                
                while has_more_pages:
                    # Fetch news articles with pagination
                    news_items, has_more_pages = await fetch_historical_ticker_news(
                        tickers=[symbol],
                        date_str=date_range,
                        items_per_page=100,
                        page=page
                    )
                    
                    total_items += len(news_items)
                    log.info(f"Retrieved {len(news_items)} news items for {symbol} (page {page})")
                    page += 1
                    
                    # Small delay between pages
                    if has_more_pages:
                        await asyncio.sleep(1)
                
                log.info(f"Total news items for {symbol}: {total_items}")
                
            except Exception as e:
                log.error(f"Error processing news for {symbol}: {e}", exc_info=True)
            
            # Small delay between symbols
            await asyncio.sleep(2)
        
        # Delay between batches
        log.info(f"Completed batch {i//batch_size + 1}")
        await asyncio.sleep(5)
    
    log.info(f"Completed crypto news backfill for all {len(symbols)} symbols")

async def aggregate_news_metrics(symbols: Optional[List[str]] = None, start_date: Optional[datetime] = None) -> None:
    """
    Aggregate news sentiment metrics for each symbol.
    
    Args:
        symbols: List of cryptocurrency symbols to aggregate. If None, uses config.BASE_SYMBOLS.
        start_date: Start date for aggregation period. If None, uses 7 days before current date.
    """
    if symbols is None:
        symbols = config.BASE_SYMBOLS
    
    # Filter symbols to only include coins under $1
    log.info("Filtering symbols to only include coins under $1")
    from data_collection.coingecko_client import fetch_coin_prices
    
    try:
        # Convert symbols to base symbols (remove USDT suffix)
        base_symbols = [s.replace('USDT', '') for s in symbols]
        
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
        log.info(f"After filtering, found {len(symbols)} coins with price < $1 for metrics aggregation")
        
        if not symbols:
            log.warning("No coins under $1 found in the provided symbol list, exiting aggregation")
            return
    except Exception as e:
        log.error(f"Error filtering low-value coins for aggregation: {e}")
        return
    
    if start_date is None:
        # Default to 7 days ago if not specified
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=config.BACKFILL_DAYS)
    else:
        end_date = datetime.now(pytz.UTC)
    
    log.info(f"Aggregating news metrics for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    
    for symbol in symbols:
        try:
            # Query to aggregate news sentiment for each day
            query = """
            INSERT INTO news_metrics (symbol, date, article_count, avg_sentiment, positive_count, negative_count, neutral_count)
            SELECT 
                symbol,
                date_trunc('day', published_at) as date,
                COUNT(*) as article_count,
                AVG(sentiment_score) as avg_sentiment,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
            FROM news_data
            WHERE symbol = $1
              AND published_at >= $2
              AND published_at <= $3
            GROUP BY symbol, date_trunc('day', published_at)
            ON CONFLICT (symbol, date)
            DO UPDATE SET
                article_count = EXCLUDED.article_count,
                avg_sentiment = EXCLUDED.avg_sentiment,
                positive_count = EXCLUDED.positive_count,
                negative_count = EXCLUDED.negative_count,
                neutral_count = EXCLUDED.neutral_count,
                updated_at = NOW()
            """
            
            # Execute query
            async with config.DB_POOL.acquire() as conn:
                result = await conn.execute(query, symbol, start_date, end_date)
                log.info(f"Aggregated news metrics for {symbol}: {result}")
        
        except Exception as e:
            log.error(f"Error aggregating news metrics for {symbol}: {e}", exc_info=True)
        
        # Small delay between operations
        await asyncio.sleep(1)
    
    log.info(f"Completed news metrics aggregation for all {len(symbols)} symbols")

async def main():
    """Main entry point for crypto news backfill script."""
    # Calculate date range for backfill
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=config.BACKFILL_DAYS)
    
    # Run backfill and aggregation
    await backfill_crypto_news(start_date=start_date)
    await aggregate_news_metrics(start_date=start_date)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Backfill process interrupted by user")
    except Exception as e:
        log.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("Crypto news backfill script finished")