#!/usr/bin/env python3
"""
Social Media Data Backfill Script - Now with async support

This script fetches and stores historical social media data
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
from data_collection.twitter_client import fetch_all_low_value_coin_tweets  # Updated to use the low-value coins function
from data_collection.reddit_client import fetch_new_subreddit_posts
from sentiment_analysis.analyzer import analyze_sentiment
from database.db_utils import async_bulk_insert, async_fetch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def backfill_twitter_data(
    symbols: List[str],
    start_date: datetime
) -> None:
    """
    Backfill Twitter data with sentiment analysis.
    Uses async patterns for improved performance.
    Specifically targets low-value coins (< $1) for more effective analysis.
    """
    log.info("Starting Twitter backfill, focusing on low-value coins")
    
    try:
        # Use the specialized function for low-value coins
        tweets = await fetch_all_low_value_coin_tweets()
        
        if tweets:
            log.info(f"Retrieved {len(tweets)} tweets about low-value coins")
            
            # Process tweets in batches for efficiency
            batch_size = 50
            for i in range(0, len(tweets), batch_size):
                batch = tweets[i:i + batch_size]
                
                # Prepare data for database
                records = []
                for tweet in batch:
                    # Extract symbol from tweet metadata
                    symbol = tweet.get('coin_symbol', 'UNKNOWN')
                    
                    record = {
                        'symbol': symbol,
                        'platform': 'twitter',
                        'content_id': tweet['tweet_id'],
                        'text_content': tweet['text'][:500],
                        'created_at': tweet['created_at'],
                        'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                        'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                        'coin_price': tweet.get('coin_price', 0),  # Store the coin price at time of collection
                        'processed_at': datetime.now(pytz.UTC)
                    }
                    records.append(record)
                
                if records:
                    await async_bulk_insert(
                        table_name='social_media_data',
                        records=records,
                        conflict_fields=['platform', 'content_id'],
                        update_fields=['like_count', 'retweet_count', 'processed_at']
                    )
                    log.debug(f"Inserted {len(records)} Twitter records for low-value coins")
            
            log.info(f"Processed {len(tweets)} tweets for low-value coins")
        else:
            log.warning("No tweets found for low-value coins")
            
    except Exception as e:
        log.error(f"Error processing Twitter data for low-value coins: {e}", exc_info=True)

async def backfill_reddit_data(
    symbols: List[str],
    start_date: datetime
) -> None:
    """
    Backfill Reddit data without explicit sentiment analysis.
    Uses async patterns for improved performance.
    """
    log.info(f"Starting Reddit backfill for {len(symbols)} symbols")
    
    rate_limiter = AsyncRateLimiter(config.REDDIT_CALLS_PER_MINUTE)
    
    for symbol in symbols:
        try:
            log.info(f"Processing Reddit data for {symbol}")
            
            await rate_limiter.wait_if_needed()
            posts = await fetch_new_subreddit_posts(
                subreddit_names=[symbol],
                post_limit_per_subreddit=100
            )
            
            if posts:
                # Process posts in batches for efficiency
                batch_size = 50
                for i in range(0, len(posts), batch_size):
                    batch = posts[i:i + batch_size]
                    
                    # Prepare data for database without explicit sentiment analysis
                    records = []
                    for post in batch:
                        record = {
                            'symbol': symbol,
                            'platform': 'reddit',
                            'content_id': post['id'],
                            'text_content': post['title'][:500],
                            'created_at': datetime.fromtimestamp(
                                post['created_utc'],
                                pytz.UTC
                            ),
                            'score': post['score'],
                            'num_comments': post['num_comments'],
                            'processed_at': datetime.now(pytz.UTC)
                        }
                        records.append(record)
                    
                    if records:
                        await async_bulk_insert(
                            table_name='social_media_data',
                            records=records,
                            conflict_fields=['platform', 'content_id'],
                            update_fields=['score', 'num_comments', 'processed_at']
                        )
                        log.debug(f"Inserted {len(records)} Reddit records for {symbol}")
                
                log.info(f"Processed {len(posts)} Reddit posts for {symbol}")
            else:
                log.warning(f"No Reddit posts found for {symbol}")
            
            # Add delay between symbols
            await asyncio.sleep(1)
            
        except Exception as e:
            log.error(f"Error processing Reddit data for {symbol}: {e}", exc_info=True)
            continue

async def aggregate_social_metrics(
    symbols: List[str],
    start_date: datetime
) -> None:
    """
    Aggregate social media metrics and store results.
    """
    log.info("Starting social metrics aggregation")
    
    try:
        for symbol in symbols:
            # Query for aggregated metrics
            query = """
            SELECT 
                platform,
                COUNT(*) as post_count,
                AVG(sentiment_score) as avg_sentiment,
                AVG(sentiment_magnitude) as avg_magnitude,
                MAX(created_at) as latest_post
            FROM social_media_data
            WHERE symbol = $1 
            AND created_at >= $2
            GROUP BY platform
            """
            
            rows = await async_fetch(query, symbol, start_date)
            
            if rows:
                # Store aggregated metrics
                for row in rows:
                    metrics = {
                        'symbol': symbol,
                        'platform': row['platform'],
                        'post_count': row['post_count'],
                        'avg_sentiment': float(row['avg_sentiment'] or 0),  # Handle potential None values
                        'avg_magnitude': float(row['avg_magnitude'] or 0),  # Handle potential None values
                        'latest_post': row['latest_post'],
                        'period_start': start_date,
                        'period_end': datetime.now(pytz.UTC),
                        'processed_at': datetime.now(pytz.UTC)
                    }
                    
                    await async_bulk_insert([metrics], 'social_media_metrics')
                    
            log.info(f"Aggregated metrics for {symbol}")
            
    except Exception as e:
        log.error(f"Error aggregating social metrics: {e}", exc_info=True)

async def backfill_social_data(symbols: Optional[List[str]] = None, start_date: Optional[datetime] = None) -> None:
    """
    Main function to run social media backfill, made accessible for external imports.
    
    Args:
        symbols: List of cryptocurrency symbols to backfill. If None, uses config.BASE_SYMBOLS.
        start_date: Start date for backfill period. If None, uses config.BACKFILL_DAYS ago.
    """
    if not (hasattr(config, 'TWITTER_BEARER_TOKEN') and config.TWITTER_BEARER_TOKEN):
        log.error("Twitter API credentials not configured")
        return

    # Get target symbols
    if symbols is None:
        symbols = config.BASE_SYMBOLS
    
    if not symbols:
        log.error("No symbols configured for backfill")
        return
        
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

    # Set date range using config.BACKFILL_DAYS for consistency with other scripts
    if start_date is None:
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=config.BACKFILL_DAYS)
        log.info(f"Using default backfill period of {config.BACKFILL_DAYS} days")
    else:
        log.info(f"Using provided start date: {start_date}")

    # Run backfills concurrently
    await asyncio.gather(
        backfill_twitter_data(symbols, start_date),
        backfill_reddit_data(symbols, start_date)
    )
    
    # Run aggregation after backfills complete
    await aggregate_social_metrics(symbols, start_date)

async def main():
    """Entry point for direct script execution."""
    await backfill_social_data()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Backfill process interrupted by user")
    except Exception as e:
        log.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        log.info("Social media backfill script finished")