# /orchestration/scheduler.py

import logging
import asyncio
import threading
import time
import schedule
from typing import List, Dict, Any
import datetime
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from data_collection.api_data import fetch_market_sentiment, fetch_crypto_health_index, fetch_coin_metrics
from data_collection.binance_client import fetch_and_store_recent_klines
from data_collection.twitter_client import fetch_new_tweets, fetch_influencer_tweets, fetch_all_low_value_coin_tweets
from data_collection.reddit_client import fetch_new_subreddit_posts
from data_collection.cryptonews_client import fetch_ticker_news
from sentiment_analysis.advanced_sentiment import analyze_low_value_coin_sentiment
from utils.rate_limiter import AsyncRateLimiter
from modeling import predictor
from trading import portfolio_manager

log = logging.getLogger(__name__)

# Synchronous scheduler functions for compatibility with main.py
def start_scheduler():
    """
    Start the synchronous scheduler using the schedule library.
    This function is called from main.py and coordinates all periodic tasks.
    It runs in the main thread and doesn't return (infinite loop).
    """
    log.info("Starting synchronous scheduler")
    
    # Schedule data collection jobs
    schedule.every(5).minutes.do(run_price_collection_job)
    schedule.every(15).minutes.do(run_sentiment_collection_job)
    schedule.every(1).hours.do(run_health_metrics_job)
    schedule.every(1).hours.do(run_market_metrics_job)
    
    # Schedule sentiment analysis and low value coin jobs
    schedule.every(30).minutes.do(run_low_value_tweets_job)
    schedule.every(1).hours.do(run_low_value_sentiment_job)
    
    # Schedule model training and prediction jobs
    schedule.every(4).hours.do(run_model_training_job)
    schedule.every(1).hours.do(run_model_prediction_job)
    
    # Schedule trading jobs (paper trading by default)
    schedule.every(15).minutes.do(run_trading_job)
    
    # Run immediately on startup
    run_initial_data_collection()
    
    # Start the scheduler loop
    log.info("Scheduler started. Running scheduled jobs...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            log.error(f"Error in scheduler loop: {e}", exc_info=True)
            time.sleep(10)  # Wait a bit longer on error

def run_initial_data_collection():
    """Run initial data collection tasks on startup"""
    log.info("Running initial data collection")
    try:
        # Run in a separate thread to not block the main thread
        thread = threading.Thread(target=_run_async_job, args=(run_async_data_collection,))
        thread.daemon = True
        thread.start()
    except Exception as e:
        log.error(f"Failed to run initial data collection: {e}", exc_info=True)

def run_price_collection_job():
    """Run the price data collection job"""
    log.info("Running price data collection job")
    _run_async_job(run_async_price_collection)

def run_sentiment_collection_job():
    """Run the sentiment data collection job"""
    log.info("Running sentiment collection job")
    _run_async_job(run_async_sentiment_collection)

def run_health_metrics_job():
    """Run the health metrics collection job"""
    log.info("Running health metrics collection job")
    _run_async_job(run_async_health_collection)

def run_market_metrics_job():
    """Run the market metrics collection job"""
    log.info("Running market metrics collection job")
    _run_async_job(run_async_market_collection)

def run_low_value_tweets_job():
    """Run the low value tweets collection job"""
    log.info("Running low value tweets collection job")
    _run_async_job(run_async_low_value_tweets_collection)

def run_low_value_sentiment_job():
    """Run the low value sentiment analysis job"""
    log.info("Running low value sentiment analysis job")
    _run_async_job(run_async_low_value_sentiment_analysis)

def run_model_training_job():
    """Run the model training job"""
    log.info("Running model training job")
    try:
        predictor.train_models()
        log.info("Model training completed")
    except Exception as e:
        log.error(f"Error in model training job: {e}", exc_info=True)

def run_model_prediction_job():
    """Run the model prediction job"""
    log.info("Running model prediction job")
    try:
        predictor.generate_predictions()
        log.info("Model predictions generated")
    except Exception as e:
        log.error(f"Error in model prediction job: {e}", exc_info=True)

def run_trading_job():
    """Run the trading job"""
    log.info("Running trading job")
    try:
        portfolio_manager.execute_trading_strategy()
        log.info("Trading job completed")
    except Exception as e:
        log.error(f"Error in trading job: {e}", exc_info=True)

def _run_async_job(async_func):
    """Helper to run async functions from synchronous context"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_func())
        loop.close()
    except Exception as e:
        log.error(f"Error in async job execution: {e}", exc_info=True)

# Async functions that will be called by the synchronous scheduler
async def run_async_data_collection():
    """Run all data collection tasks"""
    scheduler = DataCollectionScheduler()
    await scheduler.collect_all_data()

async def run_async_price_collection():
    """Run price data collection for all symbols"""
    scheduler = DataCollectionScheduler()
    tasks = []
    for symbol in config.BASE_SYMBOLS:
        tasks.append(scheduler.collect_price_data(symbol))
    await asyncio.gather(*tasks)

async def run_async_sentiment_collection():
    """Run sentiment data collection for all symbols"""
    scheduler = DataCollectionScheduler()
    tasks = []
    for symbol in config.BASE_SYMBOLS:
        tasks.append(scheduler.collect_sentiment_data(symbol))
    await asyncio.gather(*tasks)

async def run_async_health_collection():
    """Run health data collection for all symbols"""
    scheduler = DataCollectionScheduler()
    tasks = []
    for symbol in config.BASE_SYMBOLS:
        tasks.append(scheduler.collect_health_data(symbol))
    await asyncio.gather(*tasks)

async def run_async_market_collection():
    """Run market data collection for all symbols"""
    scheduler = DataCollectionScheduler()
    tasks = []
    for symbol in config.BASE_SYMBOLS:
        tasks.append(scheduler.collect_market_data(symbol))
    await asyncio.gather(*tasks)

async def run_async_low_value_tweets_collection():
    """Run low value tweets collection"""
    scheduler = DataCollectionScheduler()
    await scheduler.collect_all_low_value_coin_tweets()

async def run_async_low_value_sentiment_analysis():
    """Run low value sentiment analysis"""
    scheduler = DataCollectionScheduler()
    await scheduler.analyze_low_value_coin_sentiment_data()


class DataCollectionScheduler:
    def __init__(self):
        # Initialize rate limiters
        self.cryptopanic_limiter = AsyncRateLimiter(config.CRYPTOPANIC_CALLS_PER_MINUTE)
        self.alphavantage_limiter = AsyncRateLimiter(config.ALPHAVANTAGE_CALLS_PER_MINUTE)
        self.coingecko_limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)
        self.twitter_limiter = AsyncRateLimiter(config.TWITTER_CALLS_PER_MINUTE)
        self.reddit_limiter = AsyncRateLimiter(config.REDDIT_CALLS_PER_MINUTE)
        
        # Track last collection times
        self.last_collection = {}
    
    async def collect_price_data(self, symbol: str) -> None:
        """Collect and store market price data."""
        try:
            # Fetch and store recent klines
            await fetch_and_store_recent_klines(
                symbol=symbol,
                interval="5m",
                limit=100
            )
            log.debug(f"Collected price data for {symbol}")
        except Exception as e:
            log.error(f"Error collecting price data for {symbol}: {e}")

    async def collect_sentiment_data(self, symbol: str) -> None:
        """Collect and store sentiment data."""
        try:
            # Collect CryptoPanic sentiment
            await self.cryptopanic_limiter.wait_if_needed()
            sentiment_data = await fetch_market_sentiment([symbol])
            log.debug(f"Collected CryptoPanic sentiment for {symbol}")

            # Collect Twitter data
            await self.twitter_limiter.wait_if_needed()
            tweets = await fetch_new_tweets([symbol], ["crypto", "price"])
            log.debug(f"Collected {len(tweets)} tweets for {symbol}")

            # Collect Reddit data
            await self.reddit_limiter.wait_if_needed()
            posts = await fetch_new_subreddit_posts(config.TARGET_SUBREDDITS)
            log.debug(f"Collected {len(posts)} Reddit posts")

            # Collect CryptoNews data
            news = await fetch_ticker_news([symbol])
            log.debug(f"Collected {len(news)} news items for {symbol}")

        except Exception as e:
            log.error(f"Error collecting sentiment data for {symbol}: {e}")

    async def collect_health_data(self, symbol: str) -> None:
        """Collect and store health metrics."""
        try:
            await self.alphavantage_limiter.wait_if_needed()
            health_data = await fetch_crypto_health_index(symbol)
            if health_data:
                log.debug(f"Collected health data for {symbol}")
        except Exception as e:
            log.error(f"Error collecting health data for {symbol}: {e}")

    async def collect_market_data(self, symbol: str) -> None:
        """Collect and store market metrics."""
        try:
            await self.coingecko_limiter.wait_if_needed()
            market_data = await fetch_coin_metrics(symbol)
            if market_data:
                log.debug(f"Collected market data for {symbol}")
        except Exception as e:
            log.error(f"Error collecting market data for {symbol}: {e}")

    async def collect_all_low_value_coin_tweets(self) -> None:
        """Collect tweets from all Twitter users about cryptocurrencies valued under $1."""
        try:
            # Use rate limiter to avoid hitting Twitter API limits
            await self.twitter_limiter.wait_if_needed()
            
            # Fetch tweets from all users about low-value coins
            tweets = await fetch_all_low_value_coin_tweets()
            
            if tweets:
                log.info(f"Collected {len(tweets)} tweets from all users about coins valued under $1")
            else:
                log.debug("No tweets found about coins valued under $1")
                
        except Exception as e:
            log.error(f"Error collecting tweets about low-value coins: {e}", exc_info=True)

    async def analyze_low_value_coin_sentiment_data(self) -> None:
        """Analyze sentiment specifically for cryptocurrencies valued under $1."""
        try:
            # Run the specialized low-value coin sentiment analysis
            results = await analyze_low_value_coin_sentiment()
            
            if results:
                coin_metrics = results.get('coin_metrics', {})
                cross_metrics = results.get('cross_coin_metrics', {})
                
                # Log the number of coins analyzed
                num_coins = len(coin_metrics)
                log.info(f"Analyzed sentiment for {num_coins} low-value coins")
                
                # Log the most positive and negative coins if available
                if cross_metrics:
                    most_positive = cross_metrics.get('most_positive_coin')
                    most_negative = cross_metrics.get('most_negative_coin')
                    if most_positive:
                        log.info(f"Most positive sentiment: {most_positive}")
                    if most_negative:
                        log.info(f"Most negative sentiment: {most_negative}")
                        
                    # Log positive coins count
                    positive_count = cross_metrics.get('positive_coin_count', 0)
                    if positive_count > 0:
                        log.info(f"Found {positive_count} coins with positive sentiment")
            else:
                log.debug("No sentiment data for low-value coins")
                
        except Exception as e:
            log.error(f"Error analyzing low-value coin sentiment: {e}", exc_info=True)

    async def collect_all_data(self) -> None:
        """Collect all types of data for supported symbols."""
        symbols = config.BASE_SYMBOLS
        collection_time = datetime.datetime.now(pytz.UTC)

        # Create tasks for each symbol
        tasks = []
        for symbol in symbols:
            # Only collect data types that are due based on their intervals
            if self._should_collect('price', symbol, collection_time):
                tasks.append(self.collect_price_data(symbol))
            if self._should_collect('sentiment', symbol, collection_time):
                tasks.append(self.collect_sentiment_data(symbol))
            if self._should_collect('health', symbol, collection_time):
                tasks.append(self.collect_health_data(symbol))
            if self._should_collect('market', symbol, collection_time):
                tasks.append(self.collect_market_data(symbol))

        # Add tweet collection for low-value coins (from all users) if it's time
        if self._should_collect('low_value_tweets', 'ALL', collection_time):
            tasks.append(self.collect_all_low_value_coin_tweets())
            
        # Add sentiment analysis for low-value coins if it's time
        if self._should_collect('low_value_sentiment', 'ALL', collection_time):
            tasks.append(self.analyze_low_value_coin_sentiment_data())

        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        self._update_collection_times(collection_time)

    def _should_collect(self, data_type: str, symbol: str, current_time: datetime.datetime) -> bool:
        """Check if data collection is due based on intervals."""
        last_time = self.last_collection.get(f"{data_type}_{symbol}")
        if not last_time:
            return True

        intervals = {
            'price': datetime.timedelta(minutes=5),
            'sentiment': datetime.timedelta(minutes=15),
            'health': datetime.timedelta(hours=1),
            'market': datetime.timedelta(hours=1),
            'influencer_tweets': datetime.timedelta(minutes=15),  # General influencer tweets every 15 minutes
            'low_value_tweets': datetime.timedelta(minutes=30),   # Low-value coin tweets every 30 minutes
            'low_value_sentiment': datetime.timedelta(hours=1)    # Low-value coin sentiment analysis hourly
        }
        return (current_time - last_time) >= intervals[data_type]

    def _update_collection_times(self, collection_time: datetime.datetime) -> None:
        """Update last collection times."""
        for symbol in config.BASE_SYMBOLS:
            for data_type in ['price', 'sentiment', 'health', 'market']:
                self.last_collection[f"{data_type}_{symbol}"] = collection_time
        
        # Update special collection times
        self.last_collection["influencer_tweets_ALL"] = collection_time
        self.last_collection["low_value_tweets_ALL"] = collection_time
        self.last_collection["low_value_sentiment_ALL"] = collection_time

    async def start(self):
        """Start the data collection scheduler."""
        log.info("Starting data collection scheduler")
        while True:
            try:
                await self.collect_all_data()
                await asyncio.sleep(300)  # Wait 5 minutes between collection cycles
            except Exception as e:
                log.error(f"Error in scheduler main loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

async def main():
    """Main function to run the scheduler."""
    scheduler = DataCollectionScheduler()
    await scheduler.start()

if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())

