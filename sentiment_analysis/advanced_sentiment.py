"""
Advanced Sentiment Analysis Module - Now with async support

This module provides advanced sentiment analysis features including:
- Entity recognition and sentiment
- Topic modeling
- Temporal sentiment patterns
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import asyncio
from collections import defaultdict
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from concurrent.futures import ThreadPoolExecutor

import config
from database.db_utils import async_bulk_insert
from .analyzer import analyze_sentiment, analyze_texts_batch

log = logging.getLogger(__name__)

# Initialize thread pool for CPU-bound NLP tasks
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    log.warning("Downloading spaCy model 'en_core_web_sm'...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def _process_entities_sync(text: str) -> List[Dict[str, Any]]:
    """Synchronous entity extraction function that runs in thread pool."""
    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    except Exception as e:
        log.error(f"Error extracting entities: {e}")
        return []

async def analyze_entity_sentiment(text: str) -> Optional[Dict[str, Any]]:
    """
    Analyze sentiment for named entities in text.
    Returns entity-level sentiment analysis.
    """
    if not text or len(text.strip()) == 0:
        return None
    
    try:
        # Extract entities in thread pool
        loop = asyncio.get_running_loop()
        entities = await loop.run_in_executor(
            _thread_pool,
            _process_entities_sync,
            text
        )
        
        if not entities:
            return None
        
        # Analyze sentiment for each entity's context
        entity_sentiments = []
        for entity in entities:
            # Get context window around entity
            start = max(0, entity['start'] - 100)
            end = min(len(text), entity['end'] + 100)
            context = text[start:end]
            
            # Analyze sentiment of context
            sentiment = await analyze_sentiment(context)
            if sentiment:
                entity_sentiment = {
                    'entity': entity['text'],
                    'entity_type': entity['label'],
                    'sentiment_score': sentiment['score'],
                    'sentiment_magnitude': sentiment['magnitude']
                }
                entity_sentiments.append(entity_sentiment)
        
        if entity_sentiments:
            result = {
                'text': text[:500],
                'entities': entity_sentiments,
                'timestamp': datetime.now(pytz.UTC)
            }
            await async_bulk_insert([result], 'entity_sentiment')
            return result
        
        return None
        
    except Exception as e:
        log.error(f"Error in entity sentiment analysis: {e}")
        return None

def _extract_topics_sync(texts: List[str], num_topics: int = 5) -> Optional[Dict[str, Any]]:
    """Synchronous topic modeling function that runs in thread pool."""
    try:
        # Create TF-IDF representation
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        tfidf = vectorizer.fit_transform(texts)
        
        # Apply LDA for topic modeling
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42
        )
        lda.fit(tfidf)
        
        # Get top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [
                feature_names[i] 
                for i in topic.argsort()[:-10 - 1:-1]
            ]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words
            })
        
        return {
            'topics': topics,
            'doc_topics': lda.transform(tfidf).tolist()
        }
        
    except Exception as e:
        log.error(f"Error in topic modeling: {e}")
        return None

async def analyze_topic_sentiment(
    texts: List[str],
    num_topics: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Perform topic modeling and analyze sentiment by topic.
    Returns topic-level sentiment analysis.
    """
    if not texts:
        return None
    
    try:
        # Extract topics in thread pool
        loop = asyncio.get_running_loop()
        topic_data = await loop.run_in_executor(
            _thread_pool,
            _extract_topics_sync,
            texts,
            num_topics
        )
        
        if not topic_data:
            return None
        
        # Calculate sentiment for each text
        sentiments = await analyze_texts_batch(texts)
        if not sentiments:
            return None
        
        # Calculate average sentiment by topic
        topic_sentiments = []
        doc_topics = topic_data['doc_topics']
        
        for topic_idx, topic in enumerate(topic_data['topics']):
            # Get sentiment scores weighted by topic probability
            weighted_sentiments = [
                sent['score'] * probs[topic_idx]
                for sent, probs in zip(sentiments, doc_topics)
                if probs[topic_idx] > 0.2  # Topic probability threshold
            ]
            
            if weighted_sentiments:
                avg_sentiment = float(np.mean(weighted_sentiments))
                topic_sentiment = {
                    'topic_id': topic_idx,
                    'words': topic['words'],
                    'avg_sentiment': avg_sentiment,
                    'num_docs': len(weighted_sentiments)
                }
                topic_sentiments.append(topic_sentiment)
        
        if topic_sentiments:
            result = {
                'topic_sentiments': topic_sentiments,
                'timestamp': datetime.now(pytz.UTC)
            }
            await async_bulk_insert([result], 'topic_sentiment')
            return result
        
        return None
        
    except Exception as e:
        log.error(f"Error in topic sentiment analysis: {e}")
        return None

async def analyze_temporal_patterns(
    texts: List[str],
    timestamps: List[datetime],
    window_hours: int = 24
) -> Optional[Dict[str, Any]]:
    """
    Analyze sentiment patterns over time.
    Returns temporal sentiment metrics.
    """
    if not texts or not timestamps or len(texts) != len(timestamps):
        return None
    
    try:
        # Analyze sentiment for all texts
        sentiments = await analyze_texts_batch(texts)
        if not sentiments:
            return None
        
        # Group by time windows
        window_sentiments = defaultdict(list)
        for sentiment, ts in zip(sentiments, timestamps):
            window_start = ts.replace(
                minute=0, second=0, microsecond=0
            )
            window_sentiments[window_start].append(sentiment['score'])
        
        # Calculate metrics for each window
        temporal_metrics = []
        for window_start, scores in window_sentiments.items():
            metrics = {
                'window_start': window_start,
                'avg_sentiment': float(np.mean(scores)),
                'std_sentiment': float(np.std(scores)),
                'num_texts': len(scores)
            }
            temporal_metrics.append(metrics)
        
        if temporal_metrics:
            result = {
                'window_hours': window_hours,
                'temporal_metrics': temporal_metrics,
                'timestamp': datetime.now(pytz.UTC)
            }
            await async_bulk_insert([result], 'temporal_sentiment')
            return result
        
        return None
        
    except Exception as e:
        log.error(f"Error in temporal sentiment analysis: {e}")
        return None

async def analyze_low_value_coin_sentiment(batch_size: int = 10) -> Optional[Dict[str, Any]]:
    """
    Specialized sentiment analysis for cryptocurrencies valued under $1.
    This function:
    1. Retrieves tweets about low-value coins
    2. Analyzes sentiment for each coin
    3. Tracks sentiment trends for these coins
    4. Stores results for use in trading decisions
    
    Returns:
        Dict containing sentiment analysis results or None if analysis fails
    """
    try:
        from data_collection.binance_client import get_target_symbols
        from data_collection.coingecko_client import fetch_coin_prices
        from database.db_utils import async_fetchval, async_fetch
        
        # Step 1: Get current low-value coins (under $1)
        target_symbols_usdt = await get_target_symbols()
        base_symbols = list(set([s.replace('USDT', '').upper() for s in target_symbols_usdt]))
        
        coin_prices = await fetch_coin_prices(base_symbols)
        low_value_symbols = [symbol for symbol, price in coin_prices.items() if price < 1.0]
        
        if not low_value_symbols:
            log.warning("No coins valued at less than $1 found")
            return None
            
        log.info(f"Analyzing sentiment for {len(low_value_symbols)} coins valued under $1")
        
        # Step 2: Retrieve recent tweets about these coins
        results = {}
        for i in range(0, len(low_value_symbols), batch_size):
            coin_batch = low_value_symbols[i:i+batch_size]
            for symbol in coin_batch:
                # Get most recent tweets about this coin
                query = f"""
                SELECT tweet_id, text, created_at, public_metrics
                FROM twitter_data
                WHERE 
                    text ILIKE '%{symbol}%' OR
                    text ILIKE '%${symbol}%' OR
                    text ILIKE '%#{symbol}%'
                ORDER BY created_at DESC
                LIMIT 200
                """
                
                tweets = await async_fetch(query)
                
                if not tweets:
                    log.debug(f"No tweets found for {symbol}")
                    continue
                    
                # Extract text from tweets
                texts = [tweet['text'] for tweet in tweets]
                
                # Analyze sentiment
                sentiments = await analyze_texts_batch(texts)
                
                if not sentiments:
                    continue
                
                # Calculate aggregate sentiment metrics
                scores = [s['score'] for s in sentiments]
                
                # Store extra metrics from FinBERT if available
                finbert_metrics = {}
                finbert_samples = [s for s in sentiments if 'positive' in s]
                if finbert_samples:
                    finbert_metrics = {
                        'avg_positive': float(np.mean([s['positive'] for s in finbert_samples])),
                        'avg_negative': float(np.mean([s['negative'] for s in finbert_samples])),
                        'avg_neutral': float(np.mean([s['neutral'] for s in finbert_samples])),
                        'finbert_samples': len(finbert_samples)
                    }
                
                # Calculate overall metrics
                coin_result = {
                    'symbol': symbol,
                    'price': coin_prices.get(symbol, 0),
                    'avg_sentiment': float(np.mean(scores)),
                    'std_sentiment': float(np.std(scores)),
                    'max_sentiment': float(np.max(scores)),
                    'min_sentiment': float(np.min(scores)),
                    'tweet_count': len(tweets),
                    'timestamp': datetime.now(pytz.UTC),
                    **finbert_metrics  # Include FinBERT metrics if available
                }
                
                # Store results
                results[symbol] = coin_result
                
                # Store in database for historical tracking
                await async_bulk_insert([coin_result], 'low_value_coin_sentiment')
            
            # Add delay between batches
            if i + batch_size < len(low_value_symbols):
                await asyncio.sleep(1)
        
        # Step 3: Compute cross-coin metrics
        if results:
            symbols_with_data = list(results.keys())
            avg_sentiments = [r['avg_sentiment'] for r in results.values()]
            
            cross_coin_metrics = {
                'coins_analyzed': len(symbols_with_data),
                'most_positive_coin': symbols_with_data[np.argmax(avg_sentiments)],
                'most_negative_coin': symbols_with_data[np.argmin(avg_sentiments)],
                'avg_sentiment_all': float(np.mean(avg_sentiments)),
                'timestamp': datetime.now(pytz.UTC)
            }
            
            # Add positive coins data (coins with sentiment > 0.2)
            positive_coins = [
                symbol for symbol, data in results.items() 
                if data['avg_sentiment'] > 0.2
            ]
            if positive_coins:
                cross_coin_metrics['positive_coins'] = positive_coins
                cross_coin_metrics['positive_coin_count'] = len(positive_coins)
            
            # Store cross-coin metrics
            await async_bulk_insert([cross_coin_metrics], 'low_value_cross_coin_metrics')
            
            # Return combined results
            return {
                'coin_metrics': results,
                'cross_coin_metrics': cross_coin_metrics
            }
        
        return None
        
    except Exception as e:
        log.error(f"Error analyzing low-value coin sentiment: {e}", exc_info=True)
        return None

async def analyze_historical_low_value_sentiment(target_date: datetime) -> Optional[Dict[str, Any]]:
    """
    Performs historical sentiment analysis for low-value coins for a specific date.
    This is used for backfilling purposes to ensure historical data is processed 
    using the same methodology as the continuous process.
    
    Args:
        target_date: The specific date to analyze sentiment for.
                    Analysis will use data from this date's 00:00 to 23:59 UTC.
    
    Returns:
        Dict containing sentiment analysis results or None if analysis fails
    """
    try:
        from database.db_utils import async_fetch, async_execute
        
        # Calculate date range for the target date (full day)
        start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1) - timedelta(microseconds=1)
        
        log.info(f"Running historical low-value coin sentiment analysis for {start_time.date()}")
        
        # Step 1: Find which coins were low-value on this date
        # Query historical price data to identify coins valued under $1 on the target date
        # First check if the table exists
        check_table_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'price_data'
        )
        """
        
        table_exists = await async_fetch(check_table_query)
        if not table_exists or not table_exists[0]['exists']:
            log.error("Price data table does not exist, cannot analyze low-value coins")
            return None
            
        # Check if any price data exists for the target date
        check_data_query = f"""
        SELECT COUNT(*) as count
        FROM price_data
        WHERE open_time BETWEEN '{start_time.isoformat()}' AND '{end_time.isoformat()}'
        """
        
        data_exists = await async_fetch(check_data_query)
        if not data_exists or data_exists[0]['count'] == 0:
            log.warning(f"No price data found for {start_time.date()}")
            
            # Try to find the nearest day with data instead of failing
            nearest_day_query = """
            SELECT open_time::date as date
            FROM price_data
            ORDER BY ABS(EXTRACT(EPOCH FROM (open_time - $1)))
            LIMIT 1
            """
            
            nearest_day = await async_fetch(nearest_day_query, start_time)
            if not nearest_day:
                log.error("Could not find any price data in the database")
                return None
                
            log.info(f"Using nearest available date with data: {nearest_day[0]['date']}")
            new_date = nearest_day[0]['date']
            start_time = datetime.combine(new_date, datetime.min.time()).replace(tzinfo=pytz.UTC)
            end_time = start_time + timedelta(days=1) - timedelta(microseconds=1)
        
        # Try both with and without USDT suffix to catch more symbols
        price_query = f"""
        SELECT DISTINCT 
            CASE 
                WHEN symbol LIKE '%USDT' THEN SUBSTRING(symbol FROM 1 FOR LENGTH(symbol) - 4) 
                ELSE symbol 
            END as base_symbol,
            close
        FROM price_data
        WHERE open_time BETWEEN '{start_time.isoformat()}' AND '{end_time.isoformat()}'
        AND close < 1.0
        """
        
        low_value_symbols_data = await async_fetch(price_query)
        
        # Extract just the symbols from the query results
        if not low_value_symbols_data:
            log.warning(f"No coins valued under $1 found for {start_time.date()}, trying alternative approach")
            
            # Try a more generic approach to find any symbols
            all_symbols_query = f"""
            SELECT DISTINCT 
                CASE 
                    WHEN symbol LIKE '%USDT' THEN SUBSTRING(symbol FROM 1 FOR LENGTH(symbol) - 4) 
                    ELSE symbol 
                END as base_symbol,
                close
            FROM price_data
            WHERE open_time BETWEEN '{start_time.isoformat()}' AND '{end_time.isoformat()}'
            ORDER BY close ASC
            LIMIT 20
            """
            
            low_value_symbols_data = await async_fetch(all_symbols_query)
            if not low_value_symbols_data:
                log.error(f"Could not find any symbols with price data for {start_time.date()}")
                return None
            
            log.info(f"Using alternative approach: found {len(low_value_symbols_data)} lowest-valued coins")
            
        low_value_symbols = [item['base_symbol'] for item in low_value_symbols_data]
        low_value_prices = {item['base_symbol']: item['close'] for item in low_value_symbols_data}
        
        log.info(f"Found {len(low_value_symbols)} coins to analyze on {start_time.date()}: {', '.join(low_value_symbols[:5])}...")
        
        # Step 2: Retrieve tweets about these coins from the specific date
        # Check if social_media_data table exists
        check_social_table_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'social_media_data'
        )
        """
        
        social_table_exists = await async_fetch(check_social_table_query)
        if not social_table_exists or not social_table_exists[0]['exists']:
            log.warning("Social media data table does not exist, creating it")
            create_social_table = """
            CREATE TABLE IF NOT EXISTS social_media_data (
                id SERIAL PRIMARY KEY,
                platform TEXT NOT NULL,
                content_id TEXT,
                text_content TEXT,
                created_at TIMESTAMPTZ,
                metadata JSONB,
                sentiment_score NUMERIC(10, 6),
                sentiment_magnitude NUMERIC(10, 6)
            )
            """
            await async_execute(create_social_table)
        
        results = {}
        batch_size = 10
        
        for i in range(0, len(low_value_symbols), batch_size):
            coin_batch = low_value_symbols[i:i+batch_size]
            
            for symbol in coin_batch:
                # Try to find tweets for this symbol
                # Construct query to find tweets for this coin on the target date
                tweet_query = f"""
                SELECT content_id as tweet_id, text_content as text, created_at
                FROM social_media_data
                WHERE 
                    platform = 'twitter' AND
                    created_at BETWEEN '{start_time.isoformat()}' AND '{end_time.isoformat()}' AND
                    (
                        text_content ILIKE '%{symbol}%' OR
                        text_content ILIKE '%${symbol}%' OR
                        text_content ILIKE '%#{symbol}%'
                    )
                ORDER BY created_at DESC
                LIMIT 200
                """
                
                tweets = await async_fetch(tweet_query)
                
                if not tweets:
                    log.debug(f"No tweets found for {symbol} on {start_time.date()}")
                    
                    # Try searching news data as fallback
                    # First, check if the news_data table exists
                    check_news_exists = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'news_data'
                    )
                    """
                    news_table_exists = await async_fetch(check_news_exists)
                    if not news_table_exists or not news_table_exists[0]['exists']:
                        log.debug(f"News data table does not exist, skipping news fallback for {symbol}")
                        continue
                    
                    # Check the structure of the news_data table to find the correct column names
                    check_news_structure = """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'news_data'
                    """
                    news_columns = await async_fetch(check_news_structure)
                    
                    if not news_columns:
                        log.debug(f"Could not retrieve news_data table structure, skipping news fallback for {symbol}")
                        continue
                        
                    # Get all column names
                    column_names = [col['column_name'] for col in news_columns]
                    log.debug(f"News table columns: {column_names}")
                    
                    # Find the text content column - don't default to 'content' as it might not exist
                    text_column = None
                    date_column = None
                    
                    # Find the text content column
                    for possible_name in ['text', 'content', 'body', 'article_text', 'news_text', 'title', 'description']:
                        if possible_name in column_names:
                            text_column = possible_name
                            log.info(f"Found text column in news_data: {text_column}")
                            break
                            
                    # Find the date column
                    for possible_name in ['published_at', 'date', 'timestamp', 'created_at', 'publish_date']:
                        if possible_name in column_names:
                            date_column = possible_name
                            log.info(f"Found date column in news_data: {date_column}")
                            break
                    
                    # Skip if required columns weren't found
                    if not text_column or not date_column:
                        log.warning(f"Could not identify required columns in news_data table, skipping news fallback for {symbol}")
                        continue
                    
                    news_query = f"""
                    SELECT id as news_id, {text_column} as text, {date_column} as created_at
                    FROM news_data
                    WHERE 
                        {date_column} BETWEEN '{start_time.isoformat()}' AND '{end_time.isoformat()}' AND
                        (
                            {text_column} ILIKE '%{symbol}%' OR
                            {text_column} ILIKE '%${symbol}%' OR
                            {text_column} ILIKE '%#{symbol}%'
                        )
                    ORDER BY {date_column} DESC
                    LIMIT 100
                    """
                    
                    news_items = await async_fetch(news_query)
                    if not news_items:
                        continue
                        
                    log.info(f"Found {len(news_items)} news items for {symbol} instead of tweets")
                    tweets = news_items  # Use news items as fallback
                    
                # Extract text from tweets/news
                texts = [item.get('text', '') for item in tweets if item.get('text')]
                if not texts:
                    continue
                    
                # Analyze sentiment
                sentiments = await analyze_texts_batch(texts)
                
                if not sentiments:
                    continue
                
                # Calculate aggregate sentiment metrics
                scores = [s.get('score', 0) for s in sentiments if s and 'score' in s]
                if not scores:
                    continue
                
                # Store extra metrics from FinBERT if available
                finbert_metrics = {}
                finbert_samples = [s for s in sentiments if s and 'positive' in s]
                if finbert_samples:
                    finbert_metrics = {
                        'avg_positive': float(np.mean([s.get('positive', 0) for s in finbert_samples])),
                        'avg_negative': float(np.mean([s.get('negative', 0) for s in finbert_samples])),
                        'avg_neutral': float(np.mean([s.get('neutral', 0) for s in finbert_samples])),
                        'finbert_samples': len(finbert_samples)
                    }
                
                # Calculate overall metrics - use target_date as timestamp instead of now()
                coin_result = {
                    'symbol': symbol,
                    'price': low_value_prices.get(symbol, 0),
                    'avg_sentiment': float(np.mean(scores)),
                    'std_sentiment': float(np.std(scores)),
                    'max_sentiment': float(np.max(scores)),
                    'min_sentiment': float(np.min(scores)),
                    'tweet_count': len(tweets),
                    'timestamp': target_date,  # Use target date, not current time
                    **finbert_metrics  # Include FinBERT metrics if available
                }
                
                # Store results
                results[symbol] = coin_result
                
                # Ensure low_value_coin_sentiment table exists
                check_sentiment_table_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'low_value_coin_sentiment'
                )
                """
                
                sentiment_table_exists = await async_fetch(check_sentiment_table_query)
                if not sentiment_table_exists or not sentiment_table_exists[0]['exists']:
                    log.info("Creating low_value_coin_sentiment table")
                    create_sentiment_table = """
                    CREATE TABLE IF NOT EXISTS low_value_coin_sentiment (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        price NUMERIC(16, 8),
                        avg_sentiment NUMERIC(10, 6),
                        std_sentiment NUMERIC(10, 6),
                        max_sentiment NUMERIC(10, 6),
                        min_sentiment NUMERIC(10, 6),
                        tweet_count INTEGER,
                        timestamp TIMESTAMPTZ,
                        avg_positive NUMERIC(10, 6),
                        avg_negative NUMERIC(10, 6),
                        avg_neutral NUMERIC(10, 6),
                        finbert_samples INTEGER,
                        UNIQUE(symbol, timestamp)
                    )
                    """
                    await async_execute(create_sentiment_table)
                
                # Store in database with more robust handling
                try:
                    # Check if entry already exists
                    check_entry_query = f"""
                    SELECT id FROM low_value_coin_sentiment 
                    WHERE symbol = '{symbol}' AND timestamp = '{target_date.isoformat()}'
                    """
                    existing_entry = await async_fetch(check_entry_query)
                    
                    if existing_entry:
                        # Update existing entry
                        update_query = f"""
                        UPDATE low_value_coin_sentiment
                        SET 
                            price = {coin_result['price']},
                            avg_sentiment = {coin_result['avg_sentiment']},
                            std_sentiment = {coin_result['std_sentiment']},
                            max_sentiment = {coin_result['max_sentiment']},
                            min_sentiment = {coin_result['min_sentiment']},
                            tweet_count = {coin_result['tweet_count']}
                        WHERE symbol = '{symbol}' AND timestamp = '{target_date.isoformat()}'
                        """
                        await async_execute(update_query)
                    else:
                        # Insert new entry
                        columns = ', '.join(coin_result.keys())
                        placeholders = ', '.join([f"'{str(v)}'" if isinstance(v, str) else str(v) for v in coin_result.values()])
                        insert_query = f"""
                        INSERT INTO low_value_coin_sentiment ({columns})
                        VALUES ({placeholders})
                        ON CONFLICT (symbol, timestamp) DO NOTHING
                        """
                        await async_execute(insert_query)
                except Exception as e:
                    log.error(f"Error storing sentiment data for {symbol}: {e}")
                    continue
            
            # Add delay between batches
            if i + batch_size < len(low_value_symbols):
                await asyncio.sleep(1)
        
        # Step 3: Compute cross-coin metrics
        if results:
            symbols_with_data = list(results.keys())
            if not symbols_with_data:
                log.warning(f"No sentiment data available for low-value coins on {start_time.date()}")
                return None
                
            avg_sentiments = [r['avg_sentiment'] for r in results.values()]
            
            cross_coin_metrics = {
                'coins_analyzed': len(symbols_with_data),
                'most_positive_coin': symbols_with_data[np.argmax(avg_sentiments)] if avg_sentiments else None,
                'most_negative_coin': symbols_with_data[np.argmin(avg_sentiments)] if avg_sentiments else None,
                'avg_sentiment_all': float(np.mean(avg_sentiments)) if avg_sentiments else 0.0,
                'timestamp': target_date  # Use target date, not current time
            }
            
            # Add positive coins data (coins with sentiment > 0.2)
            positive_coins = [
                symbol for symbol, data in results.items() 
                if data['avg_sentiment'] > 0.2
            ]
            if positive_coins:
                cross_coin_metrics['positive_coins'] = positive_coins
                cross_coin_metrics['positive_coin_count'] = len(positive_coins)
            
            # Ensure cross-coin metrics table exists
            check_cross_table_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'low_value_cross_coin_metrics'
            )
            """
            
            cross_table_exists = await async_fetch(check_cross_table_query)
            if not cross_table_exists or not cross_table_exists[0]['exists']:
                log.info("Creating low_value_cross_coin_metrics table")
                create_cross_table = """
                CREATE TABLE IF NOT EXISTS low_value_cross_coin_metrics (
                    id SERIAL PRIMARY KEY,
                    coins_analyzed INTEGER,
                    most_positive_coin TEXT,
                    most_negative_coin TEXT,
                    avg_sentiment_all NUMERIC(10, 6),
                    positive_coins TEXT[],
                    positive_coin_count INTEGER,
                    timestamp TIMESTAMPTZ UNIQUE
                )
                """
                await async_execute(create_cross_table)
            
            # Store cross-coin metrics with more robust handling
            try:
                # Format positive_coins array for SQL
                positive_coins_sql = "ARRAY[" + ",".join([f"'{p}'" for p in positive_coins]) + "]" if positive_coins else "NULL"
                
                # Check if entry already exists
                check_entry_query = f"""
                SELECT id FROM low_value_cross_coin_metrics 
                WHERE timestamp = '{target_date.isoformat()}'
                """
                existing_entry = await async_fetch(check_entry_query)
                
                if existing_entry:
                    # Update existing entry
                    update_query = f"""
                    UPDATE low_value_cross_coin_metrics
                    SET 
                        coins_analyzed = {cross_coin_metrics['coins_analyzed']},
                        most_positive_coin = '{cross_coin_metrics['most_positive_coin']}',
                        most_negative_coin = '{cross_coin_metrics['most_negative_coin']}',
                        avg_sentiment_all = {cross_coin_metrics['avg_sentiment_all']},
                        positive_coins = {positive_coins_sql},
                        positive_coin_count = {cross_coin_metrics.get('positive_coin_count', 0)}
                    WHERE timestamp = '{target_date.isoformat()}'
                    """
                    await async_execute(update_query)
                else:
                    # Insert new entry
                    insert_query = f"""
                    INSERT INTO low_value_cross_coin_metrics 
                    (coins_analyzed, most_positive_coin, most_negative_coin, avg_sentiment_all, 
                     positive_coins, positive_coin_count, timestamp)
                    VALUES 
                    ({cross_coin_metrics['coins_analyzed']}, 
                     '{cross_coin_metrics['most_positive_coin']}', 
                     '{cross_coin_metrics['most_negative_coin']}', 
                     {cross_coin_metrics['avg_sentiment_all']}, 
                     {positive_coins_sql}, 
                     {cross_coin_metrics.get('positive_coin_count', 0)}, 
                     '{target_date.isoformat()}')
                    ON CONFLICT (timestamp) DO NOTHING
                    """
                    await async_execute(insert_query)
            except Exception as e:
                log.error(f"Error storing cross-coin metrics: {e}")
            
            # Return combined results
            log.info(f"Successfully analyzed historical low-value coin sentiment for {start_time.date()} - found {len(results)} coins with data")
            return {
                'coin_metrics': results,
                'cross_coin_metrics': cross_coin_metrics
            }
        
        log.warning(f"No analyzable data found for low-value coins on {start_time.date()}")
        return None
        
    except Exception as e:
        log.error(f"Error analyzing historical low-value coin sentiment for {target_date.date()}: {e}", exc_info=True)
        return None