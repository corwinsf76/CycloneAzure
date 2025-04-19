# /data_collection/backfill_social_7d.py

import logging
import time
import datetime
import pytz
import pandas as pd
from typing import List, Optional
import json

# Setup project path for imports if running script directly
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming backfill_social_7d.py is inside data_collection
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # Use print before logging configured if needed, or configure basic logging early
    # print(f"Added project root to sys.path: {project_root}")


# Import project modules
import config
from database import db_utils
from data_collection import binance_client, reddit_client, twitter_client
from sentiment_analysis import analyzer
# Import the sentiment job logic helper from scheduler (or define dummy)

# --- Setup Logging ---
# Configure logging centrally based on config BEFORE first use
log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # logging.FileHandler("cyclone_bot_backfill_social.log") # Optional: File logging
    ]
)
# Define the logger AFTER basicConfig is called
log = logging.getLogger(__name__)

# Import the sentiment job logic helper from scheduler
# Define dummy function first in case import fails
def run_sentiment_analysis_job():
    log_dummy = logging.getLogger(__name__) # Need logger inside dummy too
    log_dummy.warning("Sentiment analysis logic function could not be imported. Skipping analysis.")
    # In the original code, this function didn't need to return anything,
    # but if the calling code expects a count, return 0
    return 0

try:
    # --- Corrected Import Name ---
    from orchestration.scheduler import run_sentiment_analysis_job
    log.info("Successfully imported run_sentiment_analysis_job from orchestration.scheduler")
except ImportError as e:
    # Logger 'log' is now defined and can be used here
    log.error(f"Could not import run_sentiment_analysis_job from scheduler: {e}. Sentiment analysis step will be skipped. Using dummy function.")
    # The dummy function defined above will be used

# --- Backfill Functions ---

def backfill_reddit_7d():
    """Fetches recent Reddit posts (approx last 7 days) and stores them."""
    log.info("--- Starting Reddit 7-Day Backfill ---")
    try:
        # Fetching more posts increases chance of getting 7 days, but subject to limits
        post_limit = 500 # Fetch more posts per subreddit than the regular job
        posts = reddit_client.fetch_new_subreddit_posts(config.TARGET_SUBREDDITS, post_limit_per_subreddit=post_limit)

        if posts:
             # Filter for the last 7 days locally
             now_utc = datetime.datetime.now(pytz.utc)
             seven_days_ago = now_utc - pd.Timedelta(days=7)
             recent_posts = [p for p in posts if p.get('created_utc_dt') and p['created_utc_dt'] >= seven_days_ago]
             log.info(f"Filtered {len(posts)} fetched posts down to {len(recent_posts)} from the last 7 days.")

             # Prepare for DB insertion
             records_to_insert = []
             for post in recent_posts:
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
                  inserted_count = db_utils.bulk_insert_data(records_to_insert, db_utils.reddit_data, unique_column='post_id')
                  log.info(f"Reddit Backfill: Inserted {inserted_count} new posts.")
             else:
                  log.info("Reddit Backfill: No valid recent posts prepared for insertion.")
        else:
            log.info("Reddit Backfill: No posts fetched.")
    except Exception as e:
        log.error(f"Error in Reddit Backfill Job: {e}", exc_info=True)
    log.info("--- Reddit 7-Day Backfill Finished ---")


def backfill_twitter_7d():
    """Fetches recent tweets (approx last 7 days via Recent Search) and stores them."""
    log.info("--- Starting Twitter 7-Day Backfill ---")
    # Note: Twitter Basic v2 Recent Search has limits (e.g., monthly cap, requests/15min).
    # This might not fetch *all* relevant tweets if volume is high.
    try:
        symbols = binance_client.get_target_symbols()
        # Ensure base symbols are generated correctly
        base_symbols = list(set([s.replace('USDT', '').replace('BUSD','') for s in symbols if isinstance(s, str) and (s.endswith('USDT') or s.endswith('BUSD'))]))


        if not base_symbols:
            log.warning("Twitter Backfill: No base symbols found after filtering.")
            return

        all_tweets_to_insert = []
        # Consider lowering this if consistently hitting rate limits, or implement better backoff
        max_total_tweets_backfill = 5000
        results_per_page = 100 # Max per page for recent search

        # Combine symbols into batches for querying to reduce number of API calls
        query_batch_size = 10 # Adjust as needed based on query length limits
        log.info(f"Twitter Backfill: Processing {len(base_symbols)} symbols in batches of {query_batch_size}...")

        for i in range(0, len(base_symbols), query_batch_size):
             if len(all_tweets_to_insert) >= max_total_tweets_backfill:
                 log.warning(f"Twitter Backfill: Reached max total tweets limit ({max_total_tweets_backfill}). Stopping fetch.")
                 break

             symbol_batch = base_symbols[i:i+query_batch_size]
             # Use the corrected twitter_client which builds queries without cashtags
             query = twitter_client.build_twitter_query(symbols=symbol_batch, base_keywords=config.TWITTER_QUERY_KEYWORDS)
             if not query:
                 log.debug(f"Skipping empty query for batch starting at index {i}")
                 continue

             log.info(f"Twitter Backfill: Fetching tweets for query: {query}")
             # Don't use since_id for backfill - we want the last 7 days regardless of previous runs
             fetched_tweets, _ = twitter_client.search_recent_tweets(
                 query,
                 max_total_results=max_total_tweets_backfill - len(all_tweets_to_insert), # Fetch remaining quota
                 results_per_page=results_per_page,
                 since_id=None # Explicitly None for backfill
             )

             if fetched_tweets:
                 log.info(f"Twitter Backfill: Fetched {len(fetched_tweets)} tweets for batch.")
                 for tweet in fetched_tweets:
                      record = {
                          'tweet_id': tweet.get('tweet_id'),
                          'author_id': tweet.get('author_id'),
                          'text': tweet.get('text'),
                          'created_at': tweet.get('created_at'), # Use datetime object from client fix
                          'public_metrics': tweet.get('public_metrics'), # Store as JSON
                          'hashtags': tweet.get('hashtags'), # Store as list/JSON
                          'cashtags': tweet.get('cashtags') # Store as list/JSON (even though not queried)
                      }
                      if record['tweet_id'] and record['created_at']: # Ensure essential fields exist
                           # Ensure JSON serializable if DB backend is not JSON-native
                           # Use hasattr to safely check if JSON_TYPE is defined
                           is_json_type = hasattr(db_utils, 'JSON_TYPE') and db_utils.JSON_TYPE and (db_utils.JSON_TYPE.__name__ == 'JSON' or db_utils.JSON_TYPE.__name__ == 'JSONB')

                           if isinstance(record['public_metrics'], dict) and not is_json_type:
                               record['public_metrics'] = json.dumps(record['public_metrics'])
                           elif record['public_metrics'] is None:
                               record['public_metrics'] = {} if is_json_type else json.dumps({})

                           if isinstance(record['hashtags'], list) and not is_json_type:
                                record['hashtags'] = json.dumps(record['hashtags'])
                           elif record['hashtags'] is None:
                                record['hashtags'] = [] if is_json_type else json.dumps([])

                           if isinstance(record['cashtags'], list) and not is_json_type:
                                record['cashtags'] = json.dumps(record['cashtags'])
                           elif record['cashtags'] is None:
                                record['cashtags'] = [] if is_json_type else json.dumps([])

                           all_tweets_to_insert.append(record)
                      else:
                          log.warning(f"Skipping tweet due to missing ID or created_at: {tweet}")
             else:
                  log.info(f"Twitter Backfill: No tweets found for batch query: {query}")

             # Delay between batches to respect rate limits
             # Consider increasing this further if rate limits persist
             time.sleep(5) # Increased delay slightly

        if all_tweets_to_insert:
            # Use the corrected db_utils.bulk_insert_data which handles chunking
            inserted_count = db_utils.bulk_insert_data(all_tweets_to_insert, db_utils.twitter_data, unique_column='tweet_id')
            log.info(f"Twitter Backfill: Inserted/Processed {inserted_count} new tweets.")
        else:
            log.info("Twitter Backfill: No new valid tweets prepared for insertion across all symbols.")

    except Exception as e:
        log.error(f"Error in Twitter Backfill Job: {e}", exc_info=True)
    log.info("--- Twitter 7-Day Backfill Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    log.info("--- Starting Social Media 7-Day Backfill Script ---")

    # Initialize DB (ensure tables exist)
    if not db_utils.engine:
        log.critical("DB Engine not available. Exiting.")
        sys.exit(1)
    if not db_utils.init_db():
        # Logged as Error in db_utils, main logs warning and continues, but maybe exit here too
        log.critical("Database initialization failed. Exiting backfill.")
        sys.exit(1)
    log.info("Database schema initialized successfully.")

    # --- Run Backfill Tasks ---
    # 1. Reddit
    backfill_reddit_7d()

    # 2. Twitter
    backfill_twitter_7d()

    # 3. Sentiment Analysis on newly added historical data
    log.info("Running sentiment analysis on backfilled social data...")
    # --- Corrected Function Call ---
    try:
        # Call the potentially imported (or dummy) function
        # Assuming it finds unanalyzed items from the backfill and processes them
        run_sentiment_analysis_job() # Corrected name
    except Exception as sent_err:
        log.error(f"Error executing sentiment analysis job: {sent_err}", exc_info=True)

    log.info("--- Social Media 7-Day Backfill Script Finished ---")
    print("\nSocial media backfill process complete. Check logs for details and errors.")