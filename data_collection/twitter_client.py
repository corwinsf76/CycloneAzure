# /data_collection/twitter_client.py

import logging
import datetime
import pytz
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import aiohttp
import pickle
import os
import json

import config
from database.db_utils import async_bulk_insert
from data_collection.binance_client import get_target_symbols
from data_collection.coingecko_client import fetch_coin_prices, get_coin_id

log = logging.getLogger(__name__)

SINCE_ID_FILE = "twitter_since_id.pkl"

async def save_since_id(query_key: str, since_id: str):
    """Saves the since_id for a specific query to a file."""
    try:
        existing_data = {}
        if os.path.exists(SINCE_ID_FILE):
            with open(SINCE_ID_FILE, 'rb') as f:
                existing_data = pickle.load(f)
        existing_data[query_key] = since_id
        with open(SINCE_ID_FILE, 'wb') as f:
            pickle.dump(existing_data, f)
    except Exception as e:
        log.error(f"Error saving since_id: {e}")

async def load_since_id(query_key: str) -> Optional[str]:
    """Loads the since_id for a specific query from a file."""
    try:
        if os.path.exists(SINCE_ID_FILE):
            with open(SINCE_ID_FILE, 'rb') as f:
                data = pickle.load(f)
                return data.get(query_key)
    except Exception as e:
        log.error(f"Error loading since_id: {e}")
    return None

def build_twitter_query(symbols: List[str], base_keywords: Optional[List[str]] = None) -> str:
    """Build a Twitter search query from symbols and keywords."""
    if not symbols:
        return ""
    
    # Use default keywords if none provided
    keywords = base_keywords or ["crypto", "price", "market"]
    
    # Build query components
    symbol_terms = [f"(${symbol} OR #{symbol})" for symbol in symbols]
    keyword_terms = [f"({keyword})" for keyword in keywords]
    
    # Combine with OR between symbols and AND between keywords
    query = f"({' OR '.join(symbol_terms)}) ({' OR '.join(keyword_terms)})"
    return query

def build_user_query(usernames: List[str]) -> str:
    """Build a Twitter search query to fetch tweets from specific users."""
    if not usernames:
        return ""
    
    # Build query components for each username
    user_terms = [f"from:{username}" for username in usernames]
    
    # Combine with OR between usernames
    query = f"{' OR '.join(user_terms)}"
    return query

async def search_recent_tweets(
    query: str,
    since_id: Optional[str] = None,
    max_total_results: int = 100,
    results_per_page: int = 10
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Search recent tweets using Twitter API v2.
    Returns tuple of (tweets_list, newest_id).
    """
    if not config.TWITTER_BEARER_TOKEN:
        log.error("Twitter Bearer Token not configured")
        return [], None

    base_url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {config.TWITTER_BEARER_TOKEN}"}
    
    # Request tweet fields we need
    tweet_fields = "created_at,public_metrics,entities"
    
    all_tweets = []
    newest_id = None
    pagination_token = None
    remaining_results = max_total_results

    try:
        async with aiohttp.ClientSession() as session:
            while remaining_results > 0:
                params = {
                    "query": query,
                    "max_results": min(results_per_page, remaining_results),
                    "tweet.fields": tweet_fields,
                }
                
                if since_id:
                    params["since_id"] = since_id
                if pagination_token:
                    params["pagination_token"] = pagination_token

                async with session.get(base_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        log.error(f"Twitter API error: {response.status}")
                        break

                    data = await response.json()
                    tweets = data.get("data", [])
                    
                    if not tweets:
                        break
                        
                    # Update newest_id from first tweet of first page
                    if not newest_id and tweets:
                        newest_id = tweets[0]["id"]
                    
                    # Process tweets
                    for tweet in tweets:
                        processed_tweet = {
                            'tweet_id': tweet['id'],
                            'text': tweet['text'],
                            'created_at': datetime.datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                            'public_metrics': tweet.get('public_metrics', {}),
                            'hashtags': [tag['tag'] for tag in tweet.get('entities', {}).get('hashtags', [])],
                            'cashtags': [tag['tag'] for tag in tweet.get('entities', {}).get('cashtags', [])],
                            'fetched_at': datetime.datetime.now(pytz.utc)
                        }
                        all_tweets.append(processed_tweet)
                    
                    # Check for more pages
                    meta = data.get("meta", {})
                    if "next_token" not in meta:
                        break
                    
                    pagination_token = meta["next_token"]
                    remaining_results -= len(tweets)
                    
                    # Add small delay between requests
                    await asyncio.sleep(1)

    except Exception as e:
        log.error(f"Error fetching tweets: {e}", exc_info=True)
        return [], None

    # Store tweets in database
    if all_tweets:
        try:
            await async_bulk_insert(all_tweets, 'twitter_data')
        except Exception as e:
            log.error(f"Error storing tweets in database: {e}")

    return all_tweets, newest_id

async def fetch_new_tweets(symbols: List[str], base_keywords: Optional[List[str]] = None) -> List[Dict]:
    """
    Fetch new tweets for given symbols, using since_id to avoid duplicates.
    Returns list of processed tweets.
    """
    query = build_twitter_query(symbols, base_keywords)
    if not query:
        return []

    query_key = '_'.join(sorted(symbols))
    since_id = await load_since_id(query_key)
    
    tweets, newest_id = await search_recent_tweets(
        query=query,
        since_id=since_id,
        max_total_results=config.TWITTER_FETCH_LIMIT
    )
    
    if newest_id:
        await save_since_id(query_key, newest_id)
    
    return tweets

async def fetch_influencer_tweets(batch_size: int = 15) -> List[Dict]:
    """
    Fetch tweets from crypto influencers defined in config.
    Twitter API has query limits, so we process influencers in batches.
    
    Args:
        batch_size: Number of influencers to include in a single API query
        
    Returns:
        List of processed tweets from influencers
    """
    if not config.TWITTER_INFLUENCERS:
        log.warning("No Twitter influencers configured, skipping influencer tweet collection")
        return []
        
    all_tweets = []
    
    # Process influencers in batches to respect Twitter query complexity limits
    influencer_batches = [
        config.TWITTER_INFLUENCERS[i:i + batch_size] 
        for i in range(0, len(config.TWITTER_INFLUENCERS), batch_size)
    ]
    
    log.info(f"Fetching tweets from {len(config.TWITTER_INFLUENCERS)} influencers in {len(influencer_batches)} batches")
    
    for batch_idx, influencer_batch in enumerate(influencer_batches):
        # Create a unique key for this batch to track since_id
        batch_key = f"influencers_batch_{batch_idx}"
        
        # Build query for this batch of influencers
        query = build_user_query(influencer_batch)
        since_id = await load_since_id(batch_key)
        
        log.debug(f"Fetching tweets from influencer batch {batch_idx+1}/{len(influencer_batches)}")
        
        batch_tweets, newest_id = await search_recent_tweets(
            query=query,
            since_id=since_id,
            max_total_results=config.TWITTER_FETCH_LIMIT
        )
        
        if newest_id:
            await save_since_id(batch_key, newest_id)
            
        # Add batch tweets to overall results
        all_tweets.extend(batch_tweets)
        
        # Add a delay to respect rate limits
        if batch_idx < len(influencer_batches) - 1:  # Don't delay after the last batch
            await asyncio.sleep(2)
    
    log.info(f"Fetched {len(all_tweets)} total tweets from crypto influencers")
    return all_tweets

async def fetch_low_value_coin_tweets_from_influencers(max_batches: int = 5) -> List[Dict]:
    """
    Fetch tweets from crypto influencers specifically about low-value coins (< $1).
    
    This specialized function:
    1. Gets current Binance symbols and filters for coins valued under $1
    2. Generates comprehensive search terms for each coin (symbol, name, variations)
    3. Searches tweets from the configured influencers matching these coins
    4. Groups influencers in batches to respect Twitter API limits
    
    Args:
        max_batches: Maximum number of influencer batches to process (limit API usage)
        
    Returns:
        List of processed tweets about low-value coins from influencers
    """
    if not config.TWITTER_INFLUENCERS:
        log.warning("No Twitter influencers configured, skipping collection")
        return []
    
    all_tweets = []
    
    try:
        # Step 1: Get all Binance symbols and filter for coins under $1
        target_symbols_usdt = await get_target_symbols()
        if not target_symbols_usdt:
            log.error("Failed to get target symbols from Binance")
            return []
        
        log.info(f"Retrieved {len(target_symbols_usdt)} target symbols from Binance")
        
        # Extract base symbols
        base_symbols_upper = list(set([s.replace('USDT', '').upper() for s in target_symbols_usdt]))
        log.info(f"Extracted {len(base_symbols_upper)} unique base symbols")
        
        # Fetch current prices to filter coins valued at less than $1
        log.info("Fetching coin prices to filter by value...")
        coin_prices = await fetch_coin_prices(base_symbols_upper)
        
        # Filter symbols to only include coins valued at less than $1
        low_value_symbols = [symbol for symbol, price in coin_prices.items() if price < 1.0]
        
        if not low_value_symbols:
            log.warning("No coins valued at less than $1 found")
            return []
        
        log.info(f"Found {len(low_value_symbols)} coins valued under $1: {', '.join(low_value_symbols)}")
        
        # Step 2: Get coin names and variations for more comprehensive search
        # Create mapping of symbol to coin name for richer search terms
        coin_name_mapping = {}
        coin_ids = {}
        
        # Use rate limiter for CoinGecko API calls
        rate_limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)
        
        # Get coin info in batches to avoid rate limiting
        batch_size = 5
        symbol_batches = [low_value_symbols[i:i+batch_size] for i in range(0, len(low_value_symbols), batch_size)]
        
        for batch in symbol_batches:
            for symbol in batch:
                try:
                    await rate_limiter.wait_if_needed()
                    
                    # Get coin ID first
                    coin_id = await get_coin_id(symbol.lower())
                    if coin_id:
                        coin_ids[symbol] = coin_id
                        
                        # Make another API call to get full coin info including name
                        async with aiohttp.ClientSession() as session:
                            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                            params = {'localization': 'false', 'tickers': 'false', 'market_data': 'false'}
                            
                            headers = {}
                            if hasattr(config, 'COINGECKO_API_KEY') and config.COINGECKO_API_KEY:
                                headers['x-cg-pro-api-key'] = config.COINGECKO_API_KEY
                            
                            async with session.get(url, params=params, headers=headers) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    name = data.get('name', '')
                                    if name:
                                        coin_name_mapping[symbol] = name
                                        log.debug(f"Mapped {symbol} to name: {name}")
                                else:
                                    log.warning(f"Failed to get coin info for {symbol}: HTTP {response.status}")
                    else:
                        log.warning(f"Could not find CoinGecko ID for {symbol}")
                except Exception as e:
                    log.error(f"Error getting coin info for {symbol}: {e}")
            
            # Add delay between batches
            await asyncio.sleep(2)
        
        if not coin_name_mapping:
            log.warning("Could not map any coin symbols to names")
        else:
            log.info(f"Mapped {len(coin_name_mapping)} symbols to their full names")
        
        # Step 3: Process influencers in batches
        # Smaller batch size for influencers due to Twitter query complexity limits
        influencer_batch_size = 5
        influencer_batches = [
            config.TWITTER_INFLUENCERS[i:i+influencer_batch_size]
            for i in range(0, len(config.TWITTER_INFLUENCERS), influencer_batch_size)
        ]
        
        # Limit the number of batches processed if requested
        if max_batches and max_batches > 0:
            influencer_batches = influencer_batches[:max_batches]
            log.info(f"Limited to first {max_batches} batches of influencers ({min(max_batches * influencer_batch_size, len(config.TWITTER_INFLUENCERS))} total influencers)")
        
        log.info(f"Processing {len(influencer_batches)} batches of influencers")
        
        # Process each batch of influencers
        for batch_idx, influencer_batch in enumerate(influencer_batches):
            batch_key = f"low_value_influencers_batch_{batch_idx}"
            since_id = await load_since_id(batch_key)
            
            # Create user query for this batch of influencers
            users_query = build_user_query(influencer_batch)
            
            # Process each coin separately for this batch of influencers
            for symbol in low_value_symbols:
                coin_name = coin_name_mapping.get(symbol, '')
                
                # Build search terms for this coin (symbol, name, variations)
                search_terms = []
                
                # Add symbol variations (with $ cashtag, # hashtag, and plain)
                search_terms.append(f"${symbol}")
                search_terms.append(f"#{symbol}")
                search_terms.append(symbol)
                
                # Add full name if available (with # hashtag and plain)
                if coin_name:
                    search_terms.append(f"#{coin_name}")
                    search_terms.append(coin_name)
                    
                    # Add name without spaces for hashtags
                    no_space_name = coin_name.replace(" ", "")
                    if no_space_name != coin_name:
                        search_terms.append(f"#{no_space_name}")
                
                # Combine all search terms with OR
                coin_query = f"({' OR '.join(search_terms)})"
                
                # Combine user query and coin query
                query = f"{users_query} {coin_query}"
                log.debug(f"Searching for: {query}")
                
                try:
                    tweets, newest_id = await search_recent_tweets(
                        query=query,
                        since_id=since_id,
                        max_total_results=config.TWITTER_FETCH_LIMIT
                    )
                    
                    if tweets:
                        log.info(f"Found {len(tweets)} tweets about {symbol} ({coin_name}) from batch {batch_idx+1} influencers")
                        
                        # Enhance tweet metadata with coin info
                        for tweet in tweets:
                            tweet['coin_symbol'] = symbol
                            tweet['coin_name'] = coin_name
                            tweet['coin_price'] = coin_prices.get(symbol, 0)
                        
                        all_tweets.extend(tweets)
                    else:
                        log.debug(f"No tweets found about {symbol} from batch {batch_idx+1} influencers")
                    
                    # Update since_id for this batch if needed
                    if newest_id and (not since_id or newest_id > since_id):
                        since_id = newest_id
                    
                    # Small delay between coin queries
                    await asyncio.sleep(1)
                except Exception as e:
                    log.error(f"Error searching tweets for {symbol} from batch {batch_idx+1}: {e}")
            
            # Save the latest since_id for this batch
            if since_id:
                await save_since_id(batch_key, since_id)
            
            # Delay between influencer batches
            if batch_idx < len(influencer_batches) - 1:
                await asyncio.sleep(3)
        
        log.info(f"Found a total of {len(all_tweets)} tweets about coins under $1 from the configured influencers")
        
    except Exception as e:
        log.error(f"Error in fetch_low_value_coin_tweets_from_influencers: {e}", exc_info=True)
    
    return all_tweets

async def fetch_all_low_value_coin_tweets(batch_size: int = 5) -> List[Dict]:
    """
    Fetch tweets from all Twitter users about cryptocurrencies valued under $1.
    
    This specialized function:
    1. Gets current Binance symbols and filters for coins valued under $1
    2. Generates comprehensive search terms for each coin (symbol, name, variations)
    3. Searches all public tweets mentioning these low-value coins
    4. Processes coins in batches to respect Twitter API limits
    
    Args:
        batch_size: Number of coins to process in each batch (to manage API usage)
        
    Returns:
        List of processed tweets about low-value coins from all users
    """
    all_tweets = []
    
    try:
        # Step 1: Get all Binance symbols and filter for coins under $1
        target_symbols_usdt = await get_target_symbols()
        if not target_symbols_usdt:
            log.error("Failed to get target symbols from Binance")
            return []
        
        log.info(f"Retrieved {len(target_symbols_usdt)} target symbols from Binance")
        
        # Extract base symbols
        base_symbols_upper = list(set([s.replace('USDT', '').upper() for s in target_symbols_usdt]))
        log.info(f"Extracted {len(base_symbols_upper)} unique base symbols")
        
        # Fetch current prices to filter coins valued at less than $1
        log.info("Fetching coin prices to filter by value...")
        coin_prices = await fetch_coin_prices(base_symbols_upper)
        
        # Filter symbols to only include coins valued at less than $1
        low_value_symbols = [symbol for symbol, price in coin_prices.items() if price < 1.0]
        
        if not low_value_symbols:
            log.warning("No coins valued at less than $1 found")
            return []
        
        log.info(f"Found {len(low_value_symbols)} coins valued under $1: {', '.join(low_value_symbols)}")
        
        # Step 2: Get coin names and variations for more comprehensive search
        # Create mapping of symbol to coin name for richer search terms
        coin_name_mapping = {}
        coin_ids = {}
        
        # Use rate limiter for CoinGecko API calls
        rate_limiter = AsyncRateLimiter(config.COINGECKO_CALLS_PER_MINUTE)
        
        # Get coin info in batches to avoid rate limiting
        symbol_batches = [low_value_symbols[i:i+batch_size] for i in range(0, len(low_value_symbols), batch_size)]
        
        for batch in symbol_batches:
            for symbol in batch:
                try:
                    await rate_limiter.wait_if_needed()
                    
                    # Get coin ID first
                    coin_id = await get_coin_id(symbol.lower())
                    if coin_id:
                        coin_ids[symbol] = coin_id
                        
                        # Make another API call to get full coin info including name
                        async with aiohttp.ClientSession() as session:
                            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                            params = {'localization': 'false', 'tickers': 'false', 'market_data': 'false'}
                            
                            headers = {}
                            if hasattr(config, 'COINGECKO_API_KEY') and config.COINGECKO_API_KEY:
                                headers['x-cg-pro-api-key'] = config.COINGECKO_API_KEY
                            
                            async with session.get(url, params=params, headers=headers) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    name = data.get('name', '')
                                    if name:
                                        coin_name_mapping[symbol] = name
                                        log.debug(f"Mapped {symbol} to name: {name}")
                                else:
                                    log.warning(f"Failed to get coin info for {symbol}: HTTP {response.status}")
                    else:
                        log.warning(f"Could not find CoinGecko ID for {symbol}")
                except Exception as e:
                    log.error(f"Error getting coin info for {symbol}: {e}")
            
            # Add delay between batches
            await asyncio.sleep(2)
        
        if not coin_name_mapping:
            log.warning("Could not map any coin symbols to names")
        else:
            log.info(f"Mapped {len(coin_name_mapping)} symbols to their full names")
        
        # Step 3: Process each low-value coin
        for symbol in low_value_symbols:
            coin_name = coin_name_mapping.get(symbol, '')
            query_key = f"low_value_coin_{symbol.lower()}"
            since_id = await load_since_id(query_key)
            
            # Build search terms for this coin (symbol, name, variations)
            search_terms = []
            
            # Add symbol variations (with $ cashtag, # hashtag, and plain)
            search_terms.append(f"${symbol}")
            search_terms.append(f"#{symbol}")
            search_terms.append(symbol)
            
            # Add full name if available (with # hashtag and plain)
            if coin_name:
                search_terms.append(f"#{coin_name}")
                search_terms.append(coin_name)
                
                # Add name without spaces for hashtags
                no_space_name = coin_name.replace(" ", "")
                if no_space_name != coin_name:
                    search_terms.append(f"#{no_space_name}")
            
            # Combine all search terms with OR
            query = f"{' OR '.join(search_terms)}"
            log.debug(f"Searching for: {query}")
            
            try:
                tweets, newest_id = await search_recent_tweets(
                    query=query,
                    since_id=since_id,
                    max_total_results=config.TWITTER_FETCH_LIMIT
                )
                
                if tweets:
                    log.info(f"Found {len(tweets)} tweets about {symbol} ({coin_name})")
                    
                    # Enhance tweet metadata with coin info
                    for tweet in tweets:
                        tweet['coin_symbol'] = symbol
                        tweet['coin_name'] = coin_name
                        tweet['coin_price'] = coin_prices.get(symbol, 0)
                    
                    all_tweets.extend(tweets)
                else:
                    log.debug(f"No tweets found about {symbol}")
                
                # Update since_id for this coin
                if newest_id:
                    await save_since_id(query_key, newest_id)
                
                # Small delay between coin searches
                await asyncio.sleep(1)
            except Exception as e:
                log.error(f"Error searching tweets for {symbol}: {e}")
        
        log.info(f"Found a total of {len(all_tweets)} tweets about coins under $1")
        
    except Exception as e:
        log.error(f"Error in fetch_all_low_value_coin_tweets: {e}", exc_info=True)
    
    return all_tweets

async def fetch_tweets(
    query: str,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Fetch tweets for a specific query for the dashboard.
    
    Args:
        query: Search query (e.g., "#BTC OR $BTC")
        start_time: Start time for tweet search
        end_time: End time for tweet search
        limit: Maximum number of tweets to fetch
        
    Returns:
        List of tweet objects
    """
    try:
        log.info(f"Fetching tweets for query: {query}")
        
        # Convert start/end times to Twitter API format if provided
        since_id = None
        
        # Search for tweets
        tweets, _ = await search_recent_tweets(
            query=query,
            since_id=since_id,
            max_total_results=limit
        )
        
        # Filter by date if needed
        if start_time:
            tweets = [
                tweet for tweet in tweets 
                if tweet['created_at'] >= start_time
            ]
            
        if end_time:
            tweets = [
                tweet for tweet in tweets 
                if tweet['created_at'] <= end_time
            ]
        
        log.info(f"Found {len(tweets)} tweets matching query: {query}")
        return tweets
        
    except Exception as e:
        log.error(f"Error fetching tweets for query {query}: {e}")
        return []