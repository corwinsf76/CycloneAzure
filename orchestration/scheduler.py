# /orchestration/scheduler.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from data_collection.alphavantage_client import fetch_crypto_health_index
from data_collection.coingecko_client import fetch_coin_metrics
from data_collection.cryptopanic_client import fetch_news_sentiment
from database.db_utils import (
    store_cryptopanic_sentiment,
    store_alphavantage_health,
    store_coingecko_metrics
)
from utils.rate_limiter import AsyncRateLimiter
import config

log = logging.getLogger(__name__)

class DataCollectionScheduler:
    def __init__(self):
        # Initialize rate limiters with calls per minute
        self.cryptopanic_limiter = AsyncRateLimiter(config.CRYPTOPANIC_CALLS_PER_MINUTE)
        self.alphavantage_limiter = AsyncRateLimiter(config.ALPHAVANTAGE_CALLS_PER_MINUTE)
        self.coingecko_limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)
        
    async def collect_all_data(self):
        """Collect all types of data for supported symbols."""
        symbols = config.BASE_SYMBOLS
        
        # Collect data in parallel for each symbol
        tasks = []
        for symbol in symbols:
            tasks.extend([
                self.collect_sentiment_data(symbol),
                self.collect_health_data(symbol),
                self.collect_market_data(symbol)
            ])
        
        await asyncio.gather(*tasks)

    async def collect_sentiment_data(self, symbol: str) -> None:
        """Collect and store sentiment data for a symbol."""
        try:
            # Wait if needed before making API request
            await self.cryptopanic_limiter.wait_if_needed()
            sentiment_data = await fetch_news_sentiment(symbol)
            if sentiment_data:
                await store_cryptopanic_sentiment(sentiment_data)
                log.info(f"Stored CryptoPanic sentiment data for {symbol}")
        except Exception as e:
            log.error(f"Error collecting sentiment data for {symbol}: {e}")

    async def collect_health_data(self, symbol: str) -> None:
        """Collect and store health metrics for a symbol."""
        try:
            # Wait if needed before making API request
            await self.alphavantage_limiter.wait_if_needed()
            health_data = await fetch_crypto_health_index(symbol)
            if health_data:
                await store_alphavantage_health(health_data)
                log.info(f"Stored AlphaVantage health data for {symbol}")
        except Exception as e:
            log.error(f"Error collecting health data for {symbol}: {e}")

    async def collect_market_data(self, symbol: str) -> None:
        """Collect and store market metrics for a symbol."""
        try:
            # Wait if needed before making API request
            await self.coingecko_limiter.wait_if_needed()
            market_data = await fetch_coin_metrics(symbol)
            if market_data:
                await store_coingecko_metrics(market_data)
                log.info(f"Stored CoinGecko market data for {symbol}")
        except Exception as e:
            log.error(f"Error collecting market data for {symbol}: {e}")

    async def start(self):
        """Start the data collection scheduler."""
        while True:
            try:
                await self.collect_all_data()
                # Wait for 5 minutes before next collection
                await asyncio.sleep(300)
            except Exception as e:
                log.error(f"Error in scheduler main loop: {e}")
                # Wait for 1 minute before retry on error
                await asyncio.sleep(60)

async def main():
    """Main entry point for the scheduler."""
    scheduler = DataCollectionScheduler()
    await scheduler.start()

def start_scheduler():
    """
    Non-async wrapper to run the scheduler from synchronous code.
    This function will start the event loop if needed or use the existing one.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (e.g., in Jupyter), use create_task
            log.info("Using existing event loop to start scheduler")
            asyncio.create_task(main())
        else:
            # Otherwise, run until complete
            log.info("Starting new event loop for scheduler")
            loop.run_until_complete(main())
    except Exception as e:
        log.error(f"Error starting scheduler: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

