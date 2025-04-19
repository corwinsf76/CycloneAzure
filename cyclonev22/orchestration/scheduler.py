# /orchestration/scheduler.py

import logging
import schedule
import time
import datetime
import pytz
import pandas as pd
import numpy as np # <--- Added numpy import
from sqlalchemy.sql import select, desc, text # <--- Added sqlalchemy select import
from binance.client import Client # <--- Added binance Client import
from typing import Dict, Optional
import json # <--- Added json import for handling JSON data types

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils
from data_collection import binance_client, technicals, cryptonews_client, reddit_client, twitter_client
from sentiment_analysis import analyzer
# Feature generation and prediction are called within the trading cycle, not scheduled separately
from modeling import trainer # For retraining job
from trading import trader, portfolio # For trading cycle job

log = logging.getLogger(__name__)

# --- State Management (Needs Persistence) ---
# Store the 'since_id' for Twitter searches to avoid fetching duplicates
# In a real deployment, store this in a file or database, not just in memory.
TWITTER_SINCE_IDS: Dict[str, Optional[str]] = {} # Key: query or symbol, Value: since_id

# Initialize the Portfolio Manager instance
# In a real deployment, load state from DB or file on startup.
# Ensure this instance is accessible/shared correctly if dashboard/scheduler are different processes
try:
    portfolio_manager = portfolio.PortfolioManager(initial_capital=config.INITIAL_CAPITAL_USD)
except Exception as port_init_err:
     log.critical(f"Failed to initialize Portfolio Manager: {port_init_err}", exc_info=True)
     portfolio_manager = None # Handle initialization failure

# --- Job Functions ---

def run_price_collection_job():
    """Fetches price data and calculates technical indicators."""
    log.info("--- Job: Running Price Collection & TA Calculation ---")
    try:
        symbols = binance_client.get_target_symbols()
        if not symbols:
            log.warning("Price Collection: No target symbols found.")
            return

        all_price_data = []
        for symbol in symbols:
            # Fetch recent klines (e.g., last few hours to ensure TA calculation has enough data)
            # Adjust limit/start_str as needed. Fetching ~200 periods should be enough for most TAs.
            # Fetch slightly more than needed for TA calculation buffer.
            required_periods = max(config.SMA_SLOW_PERIOD, config.EMA_SLOW_PERIOD, config.RSI_PERIOD, config.MACD_SLOW_PERIOD) + 50 # Buffer
            # Use Client constants for interval
            klines_df = binance_client.fetch_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=required_periods)

            if not klines_df.empty:
                # Calculate indicators
                klines_with_ta = technicals.calculate_indicators(klines_df)
                if not klines_with_ta.empty:
                    # Prepare for DB insertion (match schema columns)
                    klines_with_ta['symbol'] = symbol
                    # Use Client constant
                    klines_with_ta['interval'] = Client.KLINE_INTERVAL_5MINUTE
                    # Select only columns present in the price_data table definition
                    db_cols = [c.name for c in db_utils.price_data.columns if c.name != 'id' and c.name != 'fetched_at']
                    # Ensure required base columns are present before selecting
                    base_cols_present = all(c in klines_with_ta.columns for c in ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'symbol', 'interval'])
                    if base_cols_present:
                        klines_to_store = klines_with_ta[[col for col in db_cols if col in klines_with_ta.columns]]
                        all_price_data.append(klines_to_store)
                    else:
                         log.warning(f"Price Collection: Base columns missing for {symbol} after TA calculation. Skipping DB store.")
            else:
                log.warning(f"Price Collection: No klines fetched for {symbol}.")
            time.sleep(0.2) # Small delay between symbols to avoid hitting rate limits aggressively

        if all_price_data:
            combined_df = pd.concat(all_price_data, ignore_index=True)
            log.info(f"Price Collection: Processed {len(combined_df)} total klines with TA for {len(symbols)} symbols.")
            # Use df_to_db for bulk upsert/append (needs careful handling of duplicates)
            # Simplest is append, relying on the UNIQUE constraint in the DB to prevent exact duplicates
            # For proper upsert (update existing), more complex logic or DB-specific features are needed.
            try:
                 # Convert NaNs to None for DB compatibility if needed by backend
                 combined_df_for_db = combined_df.replace({np.nan: None})
                 db_utils.df_to_db(combined_df_for_db, db_utils.price_data.name, if_exists='append', index=False)
                 # Note: Appending might insert older data again if fetch window overlaps.
                 # A better approach might fetch only the *latest* candle(s) or use DB upserts.
            except Exception as db_err:
                 log.error(f"Price Collection: Error writing price data to DB: {db_err}", exc_info=True)

    except Exception as e:
        log.error(f"Error in Price Collection Job: {e}", exc_info=True)
    log.info("--- Job: Price Collection & TA Calculation Finished ---")


def run_news_collection_job():
    """Fetches news data from CryptoNews API."""
    log.info("--- Job: Running News Collection ---")
    try:
        # Fetch news for target symbols (or a broader set)
        # Using target symbols focuses news relevance
        symbols = binance_client.get_target_symbols()
        base_symbols = list(set([s.replace('USDT', '') for s in symbols if s.endswith('USDT')])) # Get base symbols like BTC, ETH

        if not base_symbols:
            log.warning("News Collection: No base symbols found to fetch news for.")
            return

        # Fetch news (API might have limits on number of tickers per request)
        # Fetch in batches if needed
        news_items = cryptonews_client.fetch_ticker_news(tickers=base_symbols[:20], items_per_page=50) # Example: first 20 symbols

        if news_items:
            # Prepare data for DB insertion
            records_to_insert = []
            for item in news_items:
                 record = {
                     'source_api': item.get('source_api'),
                     'source_publisher': item.get('source_publisher'),
                     'article_id': item.get('article_id'),
                     'title': item.get('title'),
                     'text_content': item.get('text_content'),
                     'url': item.get('url'),
                     'published_at': item.get('published_at_utc'), # Use the converted UTC datetime
                     'tickers_mentioned': item.get('tickers_mentioned') # Store as list/JSON
                 }
                 # Filter out records with missing essential data (e.g., url, published_at)
                 if record['url'] and record['published_at']:
                     # Ensure tickers_mentioned is serializable if using JSON type in DB
                     if isinstance(record['tickers_mentioned'], list) and config.DATABASE_URL and 'postgresql' in config.DATABASE_URL:
                         pass # List is fine for JSONB
                     elif isinstance(record['tickers_mentioned'], list):
                          # Check if DB type is JSON before trying to dump, otherwise join
                          if db_utils.JSON_TYPE.__name__ == 'JSON' or db_utils.JSON_TYPE.__name__ == 'JSONB':
                              record['tickers_mentioned'] = json.dumps(record['tickers_mentioned']) # Convert list to JSON string
                          else: # Assume TEXT/VARCHAR
                              record['tickers_mentioned'] = ','.join(record['tickers_mentioned']) # Convert to string otherwise

                     records_to_insert.append(record)

            if records_to_insert:
                db_utils.bulk_insert_data(records_to_insert, db_utils.news_data, unique_column='url')
            else:
                log.info("News Collection: No valid news items prepared for insertion.")
        else:
            log.info("News Collection: No news items fetched.")

    except Exception as e:
        log.error(f"Error in News Collection Job: {e}", exc_info=True)
    log.info("--- Job: News Collection Finished ---")


def run_reddit_collection_job():
    """Fetches new posts from configured Reddit subreddits."""
    log.info("--- Job: Running Reddit Collection ---")
    try:
        posts = reddit_client.fetch_new_subreddit_posts(config.TARGET_SUBREDDITS)
        if posts:
             # Prepare for DB insertion
             records_to_insert = []
             for post in posts:
                  record = {
                      'post_id': post.get('post_id'),
                      'subreddit': post.get('subreddit'),
                      'title': post.get('title'),
                      'selftext': post.get('selftext'),
                      'url': post.get('url'),
                      'score': post.get('score'),
                      'num_comments': post.get('num_comments'),
                      'created_utc': post.get('created_utc_dt') # Use datetime object
                  }
                  if record['post_id']: # Ensure primary key is present
                       records_to_insert.append(record)

             if records_to_insert:
                  db_utils.bulk_insert_data(records_to_insert, db_utils.reddit_data, unique_column='post_id')
             else:
                  log.info("Reddit Collection: No valid posts prepared for insertion.")
        else:
            log.info("Reddit Collection: No new posts fetched.")
    except Exception as e:
        log.error(f"Error in Reddit Collection Job: {e}", exc_info=True)
    log.info("--- Job: Reddit Collection Finished ---")


def run_twitter_collection_job():
    """Fetches recent tweets based on target symbols and keywords."""
    log.info("--- Job: Running Twitter Collection ---")
    global TWITTER_SINCE_IDS
    try:
        symbols = binance_client.get_target_symbols()
        base_symbols = list(set([s.replace('USDT', '') for s in symbols if s.endswith('USDT')]))

        if not base_symbols:
            log.warning("Twitter Collection: No base symbols found.")
            return

        # Query construction and execution strategy:
        # Option 1: One big query for all symbols (might hit length limits or be too broad)
        # Option 2: Query per symbol (might hit rate limits faster) - Let's try per symbol for relevance
        all_tweets_to_insert = []
        max_total_tweets = 1000 # Limit total tweets per run to manage DB load/API usage

        for symbol in base_symbols:
            if len(all_tweets_to_insert) >= max_total_tweets:
                 log.warning(f"Twitter Collection: Reached max total tweets limit ({max_total_tweets}). Stopping fetch for remaining symbols.")
                 break

            query = twitter_client.build_twitter_query(symbols=[symbol], base_keywords=[]) # Query specific to symbol first
            if not query: continue

            # Get the last fetched ID for this query/symbol (needs persistent storage)
            since_id = TWITTER_SINCE_IDS.get(symbol)
            fetched_tweets, newest_id = twitter_client.search_recent_tweets(
                query=query,
                max_total_results=config.TWITTER_MAX_RESULTS_PER_FETCH,
                since_id=since_id
            )

            if fetched_tweets:
                log.info(f"Twitter Collection: Fetched {len(fetched_tweets)} tweets for query related to {symbol}.")
                for tweet in fetched_tweets:
                     record = {
                         'tweet_id': tweet.get('tweet_id'),
                         'author_id': tweet.get('author_id'),
                         'text': tweet.get('text'),
                         'created_at': tweet.get('created_at_utc'), # Use datetime object
                         'public_metrics': tweet.get('public_metrics'), # Store as JSON
                         'hashtags': tweet.get('hashtags'), # Store as list/JSON
                         'cashtags': tweet.get('cashtags') # Store as list/JSON
                     }
                     if record['tweet_id']:
                          # Ensure JSON serializable if needed
                          if isinstance(record['public_metrics'], dict) and config.DATABASE_URL and ('postgresql' in config.DATABASE_URL or 'mssql' in config.DATABASE_URL): pass # Dict should work for JSON/JSONB
                          elif isinstance(record['public_metrics'], dict): record['public_metrics'] = json.dumps(record['public_metrics']) # Convert dict to JSON string

                          if isinstance(record['hashtags'], list) and config.DATABASE_URL and ('postgresql' in config.DATABASE_URL or 'mssql' in config.DATABASE_URL): pass
                          elif isinstance(record['hashtags'], list): record['hashtags'] = json.dumps(record['hashtags']) # Convert list to JSON string

                          if isinstance(record['cashtags'], list) and config.DATABASE_URL and ('postgresql' in config.DATABASE_URL or 'mssql' in config.DATABASE_URL): pass
                          elif isinstance(record['cashtags'], list): record['cashtags'] = json.dumps(record['cashtags']) # Convert list to JSON string

                          all_tweets_to_insert.append(record)
            else:
                 log.debug(f"Twitter Collection: No new tweets found for query related to {symbol}.")


            if newest_id:
                # Update the since_id for the next run (persist this!)
                log.debug(f"Updating since_id for {symbol} to {newest_id}")
                TWITTER_SINCE_IDS[symbol] = newest_id
                # TODO: Implement persistent storage for TWITTER_SINCE_IDS

            time.sleep(1) # Small delay between symbols

        if all_tweets_to_insert:
            db_utils.bulk_insert_data(all_tweets_to_insert, db_utils.twitter_data, unique_column='tweet_id')
        else:
            log.info("Twitter Collection: No new valid tweets prepared for insertion across all symbols.")

    except Exception as e:
        log.error(f"Error in Twitter Collection Job: {e}", exc_info=True)
    log.info("--- Job: Twitter Collection Finished ---")


def run_sentiment_analysis_job():
    """Analyzes sentiment for new, unprocessed text data."""
    log.info("--- Job: Running Sentiment Analysis ---")
    try:
        # --- Determine records to analyze ---
        # Strategy: Find records in source tables (news, reddit, twitter)
        # that do NOT have a corresponding entry in sentiment_analysis_results.
        records_to_analyze = []
        limit_per_source = 500 # Limit records per run to avoid overwhelming model/memory

        with db_utils.get_db_session() as db:
            if db is None: # Check if session was created
                 log.error("Sentiment Analysis: Could not get DB session.")
                 return
            if db_utils.engine is None: # Check engine too for pd.read_sql
                 log.error("Sentiment Analysis: DB engine not configured.")
                 return

            # Find unanalyzed news
            news_query = select(
                    db_utils.news_data.c.id,
                    db_utils.news_data.c.text_content.label('text') # Use label for consistency
                ).select_from(
                    db_utils.news_data.outerjoin(db_utils.sentiment_analysis_results,
                                    db_utils.news_data.c.id == db_utils.sentiment_analysis_results.c.news_id)
                ).where(
                    db_utils.sentiment_analysis_results.c.id.is_(None), # Where no sentiment result exists
                    db_utils.news_data.c.text_content.isnot(None) # Ensure text exists
                ).limit(limit_per_source)
            unanalyzed_news = pd.read_sql(news_query, db_utils.engine) # Use engine
            if not unanalyzed_news.empty:
                unanalyzed_news['source_type'] = 'news'
                records_to_analyze.extend(unanalyzed_news.to_dict('records'))
                log.info(f"Found {len(unanalyzed_news)} unanalyzed news items.")

            # Find unanalyzed reddit posts (combine title + selftext)
            reddit_query = select(
                    db_utils.reddit_data.c.id,
                    (db_utils.reddit_data.c.title + ' ' + db_utils.reddit_data.c.selftext).label('text')
                ).select_from(
                    db_utils.reddit_data.outerjoin(db_utils.sentiment_analysis_results,
                                    db_utils.reddit_data.c.id == db_utils.sentiment_analysis_results.c.reddit_id)
                ).where(
                    db_utils.sentiment_analysis_results.c.id.is_(None)
                ).limit(limit_per_source)
            unanalyzed_reddit = pd.read_sql(reddit_query, db_utils.engine) # Use engine
            if not unanalyzed_reddit.empty:
                unanalyzed_reddit['source_type'] = 'reddit'
                records_to_analyze.extend(unanalyzed_reddit.to_dict('records'))
                log.info(f"Found {len(unanalyzed_reddit)} unanalyzed reddit items.")

            # Find unanalyzed tweets
            twitter_query = select(
                    db_utils.twitter_data.c.id,
                    db_utils.twitter_data.c.text
                ).select_from(
                    db_utils.twitter_data.outerjoin(db_utils.sentiment_analysis_results,
                                    db_utils.twitter_data.c.id == db_utils.sentiment_analysis_results.c.twitter_id)
                ).where(
                    db_utils.sentiment_analysis_results.c.id.is_(None)
                ).limit(limit_per_source)
            unanalyzed_twitter = pd.read_sql(twitter_query, db_utils.engine) # Use engine
            if not unanalyzed_twitter.empty:
                unanalyzed_twitter['source_type'] = 'twitter'
                records_to_analyze.extend(unanalyzed_twitter.to_dict('records'))
                log.info(f"Found {len(unanalyzed_twitter)} unanalyzed twitter items.")

        if not records_to_analyze:
            log.info("Sentiment Analysis: No new records found to analyze.")
            return

        log.info(f"Sentiment Analysis: Processing {len(records_to_analyze)} total items.")
        # Analyze sentiment
        sentiment_results = analyzer.analyze_sentiment_batch(records_to_analyze, text_key='text')

        # Store results
        if sentiment_results:
            records_to_insert = []
            for result in sentiment_results:
                source_id = result['source_id']
                # Find original source type to link FK correctly
                original_item = next((item for item in records_to_analyze if item['id'] == source_id), None)
                if original_item:
                    source_type = original_item['source_type']
                    record = {
                        'news_id': source_id if source_type == 'news' else None,
                        'reddit_id': source_id if source_type == 'reddit' else None,
                        'twitter_id': source_id if source_type == 'twitter' else None,
                        'model_name': config.SENTIMENT_MODEL_NAME,
                        'sentiment_label': result['sentiment_label'],
                        'sentiment_score': result['sentiment_score'],
                        'analyzed_at': result['analyzed_at']
                    }
                    records_to_insert.append(record)
                else:
                    log.warning(f"Could not find original item for sentiment result with source_id {source_id}")

            if records_to_insert:
                # Use the correct table object from db_utils
                db_utils.bulk_insert_data(records_to_insert, db_utils.sentiment_analysis_results)
            else:
                log.info("Sentiment Analysis: No valid results prepared for insertion.")
        else:
            log.warning("Sentiment Analysis: Analysis returned no results.")

    except Exception as e:
        log.error(f"Error in Sentiment Analysis Job: {e}", exc_info=True)
    log.info("--- Job: Sentiment Analysis Finished ---")


def run_trading_logic_job():
    """Runs the main trading decision and execution cycle."""
    log.info("--- Job: Running Trading Logic ---")
    global portfolio_manager # Use the global instance
    try:
        # Ensure portfolio manager is initialized
        if portfolio_manager is None:
             log.error("Portfolio Manager not initialized. Cannot run trading logic.")
             # Attempt initialization or load state here if needed
             return

        # Determine trading mode (e.g., from config, DB flag, or command line arg)
        # TODO: Implement robust way to set/get trading mode (e.g., via DB or state file)
        trading_mode = "PAPER" # Default to PAPER for safety
        log.info(f"Executing trading cycle in {trading_mode} mode.")

        trader.execute_trade_cycle(portfolio_manager, trading_mode)

    except Exception as e:
        log.error(f"Error in Trading Logic Job: {e}", exc_info=True)
    log.info("--- Job: Trading Logic Finished ---")


def run_model_retraining_job():
    """Periodically retrains the prediction model."""
    log.info("--- Job: Running Model Retraining ---")
    try:
        # Define symbols and time period for retraining
        symbols_to_train = binance_client.get_target_symbols() # Retrain on current targets
        if not symbols_to_train:
             log.warning("Model Retraining: No target symbols found.")
             return

        training_end_time = datetime.datetime.now(pytz.utc) - pd.Timedelta(minutes=30) # Use data up to 30 mins ago
        training_history = pd.Timedelta(days=180) # Use last 180 days
        target_variable_name = f'target_up_{config.PREDICTION_HORIZON_PERIODS}p'

        # --- Pipeline ---
        # 1. Load Data
        training_df = trainer.load_training_data(symbols_to_train, training_end_time, training_history)

        if training_df is not None and not training_df.empty:
            # 2. Preprocess Data
            X_train, y_train, X_test, y_test, features = trainer.preprocess_data(training_df, target_variable_name)

            # 3. Train Model
            trained_model, test_metrics = trainer.train_model(X_train, y_train, X_test, y_test)

            # 4. Save Model (Overwrites previous model)
            if trained_model and test_metrics:
                trainer.save_model(trained_model, features, test_metrics)
                log.info("Model retraining and saving process completed successfully.")
                # --- Important: Reload the model in the predictor ---
                # The predictor uses a singleton. We need to reset it so it loads the new model next time.
                log.warning("Resetting predictor's loaded model to pick up retrained version on next prediction.")
                # Needs access to predictor's global state or a reset function
                try:
                    from modeling import predictor as p
                    p._loaded_model = None
                    p._model_metadata = None
                    log.info("Predictor model cache cleared.")
                except Exception as reset_err:
                    log.error(f"Could not reset predictor state after retraining: {reset_err}")

            else:
                log.error("Model retraining failed.")
        else:
            log.error("Failed to load training data for retraining.")

    except Exception as e:
        log.critical(f"An error occurred during the model retraining job: {e}", exc_info=True)
    log.info("--- Job: Model Retraining Finished ---")


# --- Scheduler Setup ---

def start_scheduler():
    """Configures and starts the job scheduler."""
    log.info("Setting up scheduled jobs...")

    # --- Schedule Jobs ---
    schedule.every(config.PRICE_FETCH_INTERVAL).seconds.do(run_price_collection_job)
    log.info(f"Scheduled Price Collection every {config.PRICE_FETCH_INTERVAL} seconds.")

    schedule.every(config.NEWS_FETCH_INTERVAL).seconds.do(run_news_collection_job)
    log.info(f"Scheduled News Collection every {config.NEWS_FETCH_INTERVAL} seconds.")

    schedule.every(config.REDDIT_FETCH_INTERVAL).seconds.do(run_reddit_collection_job)
    log.info(f"Scheduled Reddit Collection every {config.REDDIT_FETCH_INTERVAL} seconds.")

    schedule.every(config.TWITTER_FETCH_INTERVAL).seconds.do(run_twitter_collection_job)
    log.info(f"Scheduled Twitter Collection every {config.TWITTER_FETCH_INTERVAL} seconds.")

    schedule.every(config.SENTIMENT_ANALYSIS_INTERVAL).seconds.do(run_sentiment_analysis_job)
    log.info(f"Scheduled Sentiment Analysis every {config.SENTIMENT_ANALYSIS_INTERVAL} seconds.")

    # Trading logic runs frequently, potentially aligned with price interval
    schedule.every(config.TRADING_LOGIC_INTERVAL).seconds.do(run_trading_logic_job)
    log.info(f"Scheduled Trading Logic every {config.TRADING_LOGIC_INTERVAL} seconds.")

    # Retraining runs less frequently
    schedule.every(config.MODEL_RETRAIN_INTERVAL).seconds.do(run_model_retraining_job)
    # schedule.every().day.at("01:00").do(run_model_retraining_job) # Example: Run daily at 1 AM server time
    log.info(f"Scheduled Model Retraining every {config.MODEL_RETRAIN_INTERVAL} seconds.")

    # --- Run Initial Jobs? ---
    run_immediately = False # Set to True to run jobs once at startup before scheduling
    if run_immediately:
        log.info("Running initial jobs immediately...")
        try:
            run_price_collection_job()
            run_news_collection_job()
            run_reddit_collection_job()
            run_twitter_collection_job()
            run_sentiment_analysis_job()
            # Don't run trading logic or retraining immediately usually
        except Exception as e:
             log.error(f"Error during initial job run: {e}", exc_info=True)

    # --- Main Scheduling Loop ---
    log.info("Starting main scheduling loop... Press Ctrl+C to exit.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1) # Sleep for a second to avoid high CPU usage
        except KeyboardInterrupt:
            log.info("Shutdown signal received. Exiting scheduler loop...")
            break
        except Exception as e:
             # Log unexpected errors in the loop, but keep running
             log.error(f"Unhandled exception in scheduler loop: {e}", exc_info=True)
             log.info("Scheduler loop continuing after error.")
             time.sleep(5) # Brief pause after an error

    log.info("Scheduler stopped.")

