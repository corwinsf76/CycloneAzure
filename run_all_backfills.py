#!/usr/bin/env python3
"""
Master Backfill Orchestrator Script

This script runs all backfill operations in sequence:
- Market data backfill (OHLCV data from exchanges)
- API data backfill (CryptoPanic, CoinGecko, etc.)
- Crypto news backfill
- Social media data backfill
- Sentiment analysis on backfilled data

The script uses the same configuration settings defined in config.py
to ensure consistency across all backfill operations.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import List, Optional
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Set up logging first - use DEBUG level for more verbose output
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

log.debug("Importing modules...")
try:
    import config
    log.debug(f"Loaded config. BACKFILL_DAYS: {config.BACKFILL_DAYS}")
    from backfill_data import backfill_binance_data  # Changed from backfill_market_data to backfill_binance_data
    from backfill_api_data import main as backfill_api_data_main
    from backfill_cryptonews import backfill_crypto_news, aggregate_news_metrics
    from backfill_social_7d import backfill_social_data
    from analyze_backfill_sentiment import main as analyze_backfill_sentiment_main
    from backfill_low_value_sentiment import run_low_value_sentiment_backfill
    log.debug("All modules imported successfully")
except ImportError as e:
    log.critical(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

async def run_all_backfills(
    symbols: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    run_market: bool = True,
    run_api: bool = True,
    run_news: bool = True,
    run_social: bool = True,
    run_low_value_sentiment: bool = True,
    optimize_db: bool = True
) -> None:
    """Run all backfill scripts to populate the database with historical data."""
    try:
        # Initialize metrics tracking
        from utils.backfill_monitoring import BackfillMetrics, log_backfill_progress
        metrics = BackfillMetrics("complete_backfill_process")
        
        # Start progress logging in background
        progress_task = asyncio.create_task(log_backfill_progress(metrics, interval_seconds=60))
        
        # Validate inputs and set defaults if needed
        if symbols is None:
            symbols = config.BASE_SYMBOLS[:5]  # Get first 5 symbols by default
            symbols.extend(['DOGEUSDT', 'DOTUSDT'])  # Add a couple more
            log.info(f"Using {len(symbols)} symbols from config: {symbols[:5]}...")
        
        if end_date is None:
            end_date = datetime.now(pytz.UTC)

        if start_date is None:
            days = config.BACKFILL_DAYS
            log.info(f"Using default backfill period of {days} days")
            start_date = end_date - timedelta(days=config.BACKFILL_DAYS)

        log.debug(f"Final parameters: symbols={symbols}, start_date={start_date}, end_date={end_date}")

        # If optimizing database, ensure proper table structure is in place
        if optimize_db:
            log.info("Setting up database optimizations...")
            from optimize_database import setup_database_optimizations
            optimization_result = await setup_database_optimizations(force_rebuild=False)
            if optimization_result:
                log.info("Database optimization setup completed successfully")
            else:
                log.warning("Database optimization setup had issues, continuing with backfill anyway")

        # 1. Backfill price market data (Binance)
        if run_market:
            log.info("Starting market data backfill...")
            metrics.record_symbol_start("market_data")
            try:
                await backfill_binance_data(symbols=symbols, start_dt_utc=start_date, end_dt_utc=end_date)
                log.info("Market data backfill completed successfully")
                metrics.record_symbol_end("market_data", success=True)
            except Exception as e:
                metrics.record_symbol_end("market_data", success=False)
                log.error(f"Error during market data backfill: {e}")
                # Continue with next steps even if this fails

        # 2. Backfill API data (CryptoPanic, CoinGecko)
        if run_api:
            log.info("Starting API data backfill (CryptoPanic, CoinGecko)...")
            metrics.record_symbol_start("api_data")
            try:
                await backfill_api_data_main(symbols=symbols)
                metrics.record_symbol_end("api_data", success=True)
            except Exception as e:
                metrics.record_symbol_end("api_data", success=False)
                log.error(f"Error during API data backfill: {e}")
                # Continue with next steps even if this fails

        # 3. Backfill news articles (CryptoNews, CryptoPanic)
        if run_news:
            log.info("Starting crypto news backfill...")
            metrics.record_symbol_start("news_data")
            try:
                await backfill_crypto_news(symbols=symbols, start_date=start_date)
                metrics.record_symbol_end("news_data", success=True)
            except Exception as e:
                metrics.record_symbol_end("news_data", success=False)
                log.error(f"Error during crypto news backfill: {e}")
                # Continue with next steps even if this fails

        # 4. Backfill social media data (Reddit, Twitter)
        if run_social:
            log.info("Starting social media data backfill...")
            metrics.record_symbol_start("social_data")
            try:
                await backfill_social_data(symbols=symbols, start_date=start_date)
                metrics.record_symbol_end("social_data", success=True)
            except Exception as e:
                metrics.record_symbol_end("social_data", success=False)
                log.error(f"Error during social media data backfill: {e}")
                # Continue with next steps even if this fails

        # 5. Backfill sentiment for low value coins
        if run_low_value_sentiment:
            log.info("Starting low-value coin sentiment analysis backfill...")
            metrics.record_symbol_start("low_value_sentiment")
            try:
                await run_low_value_sentiment_backfill(start_date=start_date, end_date=end_date)
                metrics.record_symbol_end("low_value_sentiment", success=True)
            except Exception as e:
                metrics.record_symbol_end("low_value_sentiment", success=False)
                log.error(f"Error during low-value coin sentiment backfill: {e}")
                # Continue with next steps

        # 6. Analyze and backfill sentiment for news data
        log.info("Starting sentiment analysis for backfilled data...")
        metrics.record_symbol_start("sentiment_analysis")
        try:
            await analyze_backfill_sentiment_main()
            metrics.record_symbol_end("sentiment_analysis", success=True)
        except Exception as e:
            metrics.record_symbol_end("sentiment_analysis", success=False)
            log.error(f"Error during sentiment analysis: {e}")

        # 7. If database optimization is enabled, perform data aggregation
        if optimize_db:
            log.info("Running data aggregation for time series optimization...")
            metrics.record_symbol_start("data_aggregation")
            try:
                from database.db_utils import get_db_pool
                from database.timeseries_storage import TimeSeriesAggregator
                
                pool = await get_db_pool()
                aggregator = TimeSeriesAggregator(pool)
                aggregation_results = await aggregator.aggregate_data()
                
                log.info(f"Data aggregation completed: {aggregation_results['hourly']} hourly records, "
                        f"{aggregation_results['daily']} daily records created")
                metrics.record_symbol_end("data_aggregation", success=True)
            except Exception as e:
                metrics.record_symbol_end("data_aggregation", success=False)
                log.error(f"Error during data aggregation: {e}")

        metrics.complete()
        metrics.log_summary()
        metrics.save_to_file("logs/metrics")
        
        # Cancel progress logging
        progress_task.cancel()
                
        log.info("All backfill operations completed")
    
    except Exception as e:
        log.error(f"Unhandled error during backfill process: {e}", exc_info=True)

def parse_args():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all backfill operations for cryptocurrency data")
    
    parser.add_argument(
        "--symbols", 
        type=str, 
        help="Comma-separated list of cryptocurrency symbols to backfill data for"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        help=f"Number of days to backfill data for (default: {config.BACKFILL_DAYS})"
    )
    
    parser.add_argument(
        "--no-market", 
        action="store_true", 
        help="Skip market data backfill"
    )
    
    parser.add_argument(
        "--no-api", 
        action="store_true", 
        help="Skip API data backfill"
    )
    
    parser.add_argument(
        "--no-news", 
        action="store_true", 
        help="Skip crypto news backfill"
    )
    
    parser.add_argument(
        "--no-social", 
        action="store_true", 
        help="Skip social media data backfill"
    )
    
    parser.add_argument(
        "--no-low-value-sentiment",
        action="store_true",
        help="Skip low-value coin sentiment analysis backfill"
    )
    
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    # Process symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Process start date
    start_date = None
    if args.days:
        start_date = datetime.now(pytz.UTC) - timedelta(days=args.days)
    
    # Run all backfills
    await run_all_backfills(
        symbols=symbols,
        start_date=start_date,
        run_market=not args.no_market,
        run_api=not args.no_api,
        run_news=not args.no_news,
        run_social=not args.no_social,
        run_low_value_sentiment=not args.no_low_value_sentiment
    )

if __name__ == "__main__":
    asyncio.run(main())