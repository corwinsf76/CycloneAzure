# /data_collection/twitter_client.py

import logging
import tweepy
import datetime
import pytz
from typing import List, Dict, Optional, Tuple, Any
import time
import json # Added for JSON serialization if needed
import os
import pickle

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
# from .. import config # Use relative import if running as part of a package

log = logging.getLogger(__name__)

# Custom exception classes
class RateLimitError(Exception):
    """Exception raised when a rate limit is exceeded."""
    pass

# File to persist since_id for polling
SINCE_ID_FILE = "twitter_since_id.pkl"

def save_since_id(query_key: str, since_id: str):
    """Saves the since_id for a specific query to a file."""
    try:
        if os.path.exists(SINCE_ID_FILE):
            with open(SINCE_ID_FILE, "rb") as f:
                since_id_store = pickle.load(f)
        else:
            since_id_store = {}

        since_id_store[query_key] = since_id

        with open(SINCE_ID_FILE, "wb") as f:
            pickle.dump(since_id_store, f)
        log.info(f"Saved since_id for query '{query_key}': {since_id}")
    except Exception as e:
        log.error(f"Error saving since_id: {e}", exc_info=True)

def load_since_id(query_key: str) -> Optional[str]:
    """Loads the since_id for a specific query from a file."""
    try:
        if os.path.exists(SINCE_ID_FILE):
            with open(SINCE_ID_FILE, "rb") as f:
                since_id_store = pickle.load(f)
            return since_id_store.get(query_key)
    except Exception as e:
        log.error(f"Error loading since_id: {e}", exc_info=True)
    return None

def split_query(query: str, max_length: int = 512) -> List[str]:
    """Splits a long query into smaller parts to fit within the Twitter API limit."""
    if len(query) <= max_length:
        return [query]

    terms = query.split(" OR ")
    queries = []
    current_query = ""

    for term in terms:
        if len(current_query) + len(term) + 4 <= max_length:  # +4 for " OR "
            current_query = f"{current_query} OR {term}" if current_query else term
        else:
            queries.append(current_query)
            current_query = term

    if current_query:
        queries.append(current_query)

    return queries

# --- Twitter Client Initialization ---
_twitter_client_v2 = None

def get_twitter_client_v2():
    """Initializes and returns the Tweepy API v2 client singleton."""
    global _twitter_client_v2
    if _twitter_client_v2 is None:
        bearer_token = config.TWITTER_BEARER_TOKEN
        if not bearer_token:
            log.error("Twitter API v2 Bearer Token (TWITTER_BEARER_TOKEN) not configured.")
            return None
        try:
            log.info("Initializing Tweepy client for Twitter API v2...")
            # Use bearer token for app-only authentication (suitable for searching tweets)
            _twitter_client_v2 = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False) # Handle rate limits manually for logging
            log.info("Tweepy client object created successfully (using Bearer Token).")

        except tweepy.TweepyException as e:
            log.error(f"TweepyException initializing Twitter client: {e}", exc_info=True)
            _twitter_client_v2 = None
        except Exception as e:
            log.error(f"Unexpected error initializing Twitter client: {e}", exc_info=True)
            _twitter_client_v2 = None
    return _twitter_client_v2

def build_twitter_query(symbols: List[str], base_keywords: List[str] = config.TWITTER_QUERY_KEYWORDS) -> str:
    """
    Builds a search query string for the Twitter API v2 Recent Search endpoint.
    Focuses on hashtags for the given symbols, combined with base keywords.
    Excludes retweets and requires English language.

    Args:
        symbols (List[str]): List of cryptocurrency symbols (e.g., ['BTC', 'ETH']).
                             Should be just the base symbol, without 'USDT'.
        base_keywords (List[str]): General keywords to include (e.g., ['crypto']).

    Returns:
        str: A formatted query string for the Twitter API.
             Returns an empty string if no symbols or keywords provided.
    """
    if not symbols and not base_keywords:
        log.warning("Cannot build Twitter query without symbols or keywords.")
        return ""

    symbol_query_parts = []
    if symbols:
        # Create hashtag parts for each symbol
        symbol_parts = []
        for symbol in symbols:
            s_upper = symbol.upper().replace('USDT', '').replace('BUSD','') # Ensure base symbol, uppercase
            if s_upper:
                # --- Modified: Removed the cashtag ($SYMBOL) part ---
                symbol_parts.append(f"#{s_upper}")
        if symbol_parts:
            # Keep the OR structure between hashtags
            symbol_query_parts.append(f"({' OR '.join(symbol_parts)})")

    keyword_query_parts = []
    if base_keywords:
        keyword_part = " ".join([f'"{k}"' if ' ' in k else k for k in base_keywords]) # Quote keywords with spaces
        keyword_query_parts.append(f"({keyword_part})")

    # Combine symbol and keyword parts (if both exist, combine with OR or AND depending on need - using OR for broader reach)
    combined_terms = " OR ".join(symbol_query_parts + keyword_query_parts)
    if not combined_terms:
         log.warning("No valid symbols or keywords resulted in query terms.")
         return ""

    # Combine parts, add language filter and exclude retweets
    full_query = f"({combined_terms}) lang:en -is:retweet"

    # Twitter query length limit is 512 chars for standard v2 basic/elevated, 1024 for academic
    max_len = 512
    if len(full_query) > max_len:
        log.warning(f"Generated Twitter query exceeds max length ({max_len}). Truncating: {full_query}")
        # Simple truncation, might break logic. Consider smarter query splitting if this happens often.
        full_query = full_query[:max_len].rsplit(' ', 1)[0] # Try to cut at last space
        log.warning(f"Truncated query: {full_query}")

    log.debug(f"Generated Twitter Query: {full_query}")
    return full_query.strip()


def search_recent_tweets(query: str, since_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Search for recent tweets."""
    client = get_twitter_client_v2()
    if not client:
        log.error("Cannot search tweets, Twitter client not available.")
        return [], None

    fetched_tweets = []
    current_newest: Optional[str] = None
    next_page_token = None
    page_num = 0
    
    try:
        while True:
            response = client.search_recent_tweets(
                query=query,
                since_id=since_id,
                next_token=next_page_token
            )
            
            if hasattr(response, 'errors') and getattr(response, 'errors'):
                log.error(f"Twitter API returned errors on page {page_num + 1}: {getattr(response, 'errors')}")
                for error in getattr(response, 'errors'):
                    if error.get('code') == 88:  # Rate limit error
                        raise RateLimitError(error.get('message', 'Rate limit exceeded'))
                break
            
            if hasattr(response, 'data') and getattr(response, 'data'):
                tweets_data = getattr(response, 'data')
                log.info(f"Received {len(tweets_data)} tweets on page {page_num + 1}.")
                fetched_tweets.extend(tweets_data)
                
                # Track newest_id for incrementally fetching newer tweets
                meta = getattr(response, 'meta', {})
                if meta and 'newest_id' in meta:
                    current_newest = meta['newest_id']
                
                if meta and 'next_token' in meta:
                    next_page_token = meta['next_token']
                    page_num += 1
                else:
                    break  # No more pages
            else:
                break  # No data in response
                
    except tweepy.TweepyException as e:
        if isinstance(e, tweepy.BadRequest):
            log.error(f"Bad request error: {str(e)}")
        elif isinstance(e, tweepy.HTTPException) and hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            log.error("Rate limit exceeded")
            raise RateLimitError("Twitter API rate limit exceeded")
        else:
            log.error(f"Twitter API error: {str(e)}")
        raise
        
    return fetched_tweets, current_newest


# --- Example Usage (for testing) ---
# Store the last fetched tweet ID for pagination testing
# In a real app, this state needs to be persisted (e.g., in DB or a state file)
_LATEST_TWEET_ID_STORE_TEST = {}

if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')

    print("--- Testing Twitter Client ---")

    if not config.TWITTER_BEARER_TOKEN:
         print("\nERROR: Twitter Bearer Token (TWITTER_BEARER_TOKEN) not configured in .env")
    else:
        # Test with a few symbols
        test_symbols = ['BTC', 'ETH'] # Use symbols likely to have recent tweets
        test_query = build_twitter_query(symbols=test_symbols, base_keywords=['price']) # Uses the corrected build_twitter_query

        if test_query:
            print(f"\nGenerated Query (No Cashtags): {test_query}")

            # --- Simulate polling with since_id ---
            query_key_test = "test_btc_eth_no_cashtag"
            # 1. First fetch
            print("\n--- Fetch 1 (No Since ID) ---")
            since_id_for_run = load_since_id(query_key_test) # Load since_id from file
            tweets1, newest_id1 = search_recent_tweets(query=test_query, since_id=since_id_for_run) # Fetch tweets
            if tweets1:
                print(f"Fetched {len(tweets1)} tweets.")
                print("First tweet:", tweets1[0]['text'][:100] + "...")
                print(f"Newest ID: {newest_id1}")
                if newest_id1:
                     save_since_id(query_key_test, newest_id1) # Save since_id to file
            else:
                print("No new tweets found in Fetch 1.")

            # Add a small delay before next fetch if testing pagination
            print("\nWaiting 5 seconds...")
            time.sleep(5)

            # 2. Second fetch (should ideally fetch 0 if no new tweets in 5s)
            print("\n--- Fetch 2 (Using Since ID from Fetch 1) ---")
            since_id_for_run = load_since_id(query_key_test) # Load since_id from file
            print(f"Using since_id: {since_id_for_run}")
            # Pass since_id correctly to search_recent_tweets
            tweets2, newest_id2 = search_recent_tweets(query=test_query, since_id=since_id_for_run)
            if tweets2:
                print(f"Fetched {len(tweets2)} tweets.") # Should likely be 0 or very few
                if newest_id2:
                     save_since_id(query_key_test, newest_id2) # Save since_id to file
            else:
                print("No new tweets found in Fetch 2 (as expected potentially).")
                print(f"Newest ID from Fetch 2 meta (if any): {newest_id2}") # newest_id_overall is returned

        else:
            print("Could not generate a valid test query.")

    print("\n--- Test Complete ---")