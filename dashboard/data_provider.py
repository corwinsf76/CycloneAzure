"""
Dashboard Data Provider Module - Now with async support

This module handles all data fetching operations for the dashboard
using async patterns for improved performance.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd
import numpy as np

from database.db_utils import get_db_pool
from data_collection.binance_client import get_market_data
from data_collection.cryptonews_client import fetch_crypto_news
from data_collection.twitter_client import fetch_tweets
from data_collection.reddit_client import fetch_reddit_posts
from sentiment_analysis.analyzer import analyze_sentiment
from .config_manager import config_manager

log = logging.getLogger(__name__)

async def get_price_data(
    symbol: str,
    interval: str = '1h',
    limit: int = 168
) -> pd.DataFrame:
    """
    Fetch price data asynchronously for the dashboard.
    Returns hourly price data by default (1 week = 168 hours).
    """
    try:
        data = await get_market_data(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if data:
            df = pd.DataFrame(data)
            df['open_time'] = pd.to_datetime(df['open_time'])
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        log.error(f"Error fetching price data: {e}")
        return pd.DataFrame()

async def get_sentiment_data(symbol: str) -> Dict[str, Any]:
    """
    Fetch and aggregate sentiment data asynchronously from multiple sources.
    """
    try:
        # Initialize results structure
        results = {
            'raw_data': {
                'news': [],
                'social': [],
                'market': []
            },
            'aggregates': {
                'news': {
                    'current': 0.0,
                    'trend': 0.0
                },
                'social': {
                    'current': 0.0,
                    'trend': 0.0
                },
                'market': {
                    'current': 0.0,
                    'trend': 0.0
                }
            }
        }
        
        # Fetch data from different sources concurrently
        start_time = datetime.now(pytz.UTC) - timedelta(days=1)
        
        news_task = fetch_crypto_news(
            symbol=symbol,
            from_date=start_time
        )
        
        tweets_task = fetch_tweets(
            query=f"#{symbol} OR ${symbol}",
            start_time=start_time
        )
        
        reddit_task = fetch_reddit_posts(
            query=symbol,
            after=start_time
        )
        
        # Wait for all tasks to complete
        news_items, tweets, reddit_posts = await asyncio.gather(
            news_task, tweets_task, reddit_task
        )
        
        # Process news sentiment
        if news_items:
            sentiment_tasks = []
            for item in news_items:
                text = f"{item['title']} {item.get('description', '')}"
                task = analyze_sentiment(text)
                sentiment_tasks.append(task)
            
            sentiments = await asyncio.gather(*sentiment_tasks)
            
            for item, sentiment in zip(news_items, sentiments):
                if sentiment:
                    results['raw_data']['news'].append({
                        'timestamp': item['published_at'],
                        'score': sentiment['score'],
                        'magnitude': sentiment['magnitude'],
                        'source': item['source']
                    })
        
        # Process social media sentiment
        social_items = []
        if tweets:
            social_items.extend([
                {'text': tweet['text'], 'platform': 'twitter', 'created_at': tweet['created_at']}
                for tweet in tweets
            ])
        
        if reddit_posts:
            social_items.extend([
                {
                    'text': f"{post['title']} {post.get('selftext', '')}",
                    'platform': 'reddit',
                    'created_at': datetime.fromtimestamp(post['created_utc'], pytz.UTC)
                }
                for post in reddit_posts
            ])
        
        if social_items:
            sentiment_tasks = []
            for item in social_items:
                task = analyze_sentiment(item['text'])
                sentiment_tasks.append(task)
            
            sentiments = await asyncio.gather(*sentiment_tasks)
            
            for item, sentiment in zip(social_items, sentiments):
                if sentiment:
                    results['raw_data']['social'].append({
                        'timestamp': item['created_at'],
                        'score': sentiment['score'],
                        'magnitude': sentiment['magnitude'],
                        'platform': item['platform']
                    })
        
        # Calculate aggregates
        for source in ['news', 'social']:
            data = results['raw_data'][source]
            if data:
                current_scores = [item['score'] for item in data]
                results['aggregates'][source]['current'] = np.mean(current_scores)
                
                if len(current_scores) > 1:
                    results['aggregates'][source]['trend'] = (
                        current_scores[-1] - current_scores[0]
                    )
        
        return results
        
    except Exception as e:
        log.error(f"Error fetching sentiment data: {e}")
        return {}

async def get_market_metrics(symbol: str) -> Dict[str, Any]:
    """
    Fetch market metrics asynchronously for the dashboard.
    """
    try:
        async with get_db_pool().acquire() as conn:
            query = """
            SELECT 
                market_cap,
                volume_24h,
                price_change_24h,
                market_rank,
                community_score
            FROM market_metrics
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            row = await conn.fetchrow(query, symbol)
            
            if row:
                return {
                    'market_cap': float(row['market_cap']),
                    'volume_24h': float(row['volume_24h']),
                    'price_change_24h': float(row['price_change_24h']),
                    'market_rank': int(row['market_rank']),
                    'community_score': float(row['community_score'])
                }
            
            return {}
            
    except Exception as e:
        log.error(f"Error fetching market metrics: {e}")
        return {}

async def get_technical_indicators(symbol: str) -> Dict[str, Any]:
    """
    Calculate technical indicators asynchronously for the dashboard.
    """
    try:
        # Fetch price data
        df = await get_price_data(symbol, interval='1h', limit=200)
        
        if df.empty:
            return {}
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Calculate Moving Averages
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()
        
        return {
            'rsi': float(rsi.iloc[-1]),
            'macd': {
                'value': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'hist': float(histogram.iloc[-1])
            },
            'moving_averages': {
                'sma_20': float(sma_20.iloc[-1]),
                'sma_50': float(sma_50.iloc[-1]),
                'sma_200': float(sma_200.iloc[-1])
            }
        }
        
    except Exception as e:
        log.error(f"Error calculating technical indicators: {e}")
        return {}

async def get_social_metrics(symbol: str) -> Dict[str, Any]:
    """
    Fetch social media metrics asynchronously for the dashboard.
    """
    try:
        async with get_db_pool().acquire() as conn:
            query = """
            WITH twitter_metrics AS (
                SELECT 
                    COUNT(*) as post_count,
                    AVG(sentiment_score) as avg_sentiment,
                    SUM(like_count + retweet_count) as total_engagement
                FROM social_media_data
                WHERE symbol = $1 
                AND platform = 'twitter'
                AND created_at >= NOW() - INTERVAL '24 hours'
            ),
            reddit_metrics AS (
                SELECT 
                    COUNT(*) as post_count,
                    AVG(sentiment_score) as avg_sentiment,
                    SUM(score) as total_score,
                    SUM(num_comments) as total_comments
                FROM social_media_data
                WHERE symbol = $1 
                AND platform = 'reddit'
                AND created_at >= NOW() - INTERVAL '24 hours'
            )
            SELECT 
                t.post_count as twitter_posts,
                t.avg_sentiment as twitter_sentiment,
                t.total_engagement as twitter_engagement,
                r.post_count as reddit_posts,
                r.avg_sentiment as reddit_sentiment,
                r.total_score as reddit_score,
                r.total_comments as reddit_comments
            FROM twitter_metrics t, reddit_metrics r
            """
            
            row = await conn.fetchrow(query, symbol)
            
            if row:
                return {
                    'twitter': {
                        'post_count': int(row['twitter_posts']),
                        'avg_sentiment': float(row['twitter_sentiment']),
                        'total_engagement': int(row['twitter_engagement'])
                    },
                    'reddit': {
                        'post_count': int(row['reddit_posts']),
                        'avg_sentiment': float(row['reddit_sentiment']),
                        'total_score': int(row['reddit_score']),
                        'total_comments': int(row['reddit_comments'])
                    }
                }
            
            return {}
            
    except Exception as e:
        log.error(f"Error fetching social metrics: {e}")
        return {}

async def get_sentiment_history(symbol: str, hours: int = 24) -> pd.DataFrame:
    """
    Fetch historical sentiment scores for a symbol from the database.
    """
    try:
        async with get_db_pool().acquire() as conn:
            query = """
            SELECT timestamp, score
            FROM sentiment_scores
            WHERE coin_symbol = $1 AND timestamp >= $2
            ORDER BY timestamp ASC
            """
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            rows = await conn.fetch(query, symbol, start_time)

            if rows:
                df = pd.DataFrame(rows, columns=['timestamp', 'score'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Ensure score is float
                df['score'] = df['score'].astype(float)
                return df
            return pd.DataFrame(columns=['timestamp', 'score'])
    except Exception as e:
        log.error(f"Error fetching sentiment history for {symbol}: {e}")
        return pd.DataFrame(columns=['timestamp', 'score'])

async def get_low_value_coins() -> List[Dict[str, Any]]:
    """
    Fetch coins currently priced at or below the low-value threshold.
    Assumes a 'coins' table with 'symbol' and 'current_price' columns.
    """
    try:
        threshold = await config_manager.get_config_value("LOW_VALUE_PRICE_THRESHOLD", default=1.0)
        async with get_db_pool().acquire() as conn:
            # Adjust table and column names if different in your schema
            query = """
            SELECT symbol, current_price, name
            FROM coins
            WHERE current_price <= $1 AND current_price > 0
            ORDER BY current_price DESC
            LIMIT 20 -- Limit results for performance
            """
            rows = await conn.fetch(query, float(threshold))
            return [dict(row) for row in rows]
    except Exception as e:
        log.error(f"Error fetching low-value coins: {e}")
        return []

async def get_portfolio_holdings() -> pd.DataFrame:
    """
    Fetch current portfolio holdings from the database.
    Assumes a 'portfolio_holdings' table with 'asset_symbol' and 'quantity'.
    """
    try:
        async with get_db_pool().acquire() as conn:
            # Adjust table and column names if different
            query = """
            SELECT asset_symbol, quantity
            FROM portfolio_holdings
            WHERE quantity > 0
            ORDER BY asset_symbol ASC
            """
            rows = await conn.fetch(query)
            if rows:
                df = pd.DataFrame(rows, columns=['Symbol', 'Quantity'])
                # Ensure Quantity is numeric
                df['Quantity'] = pd.to_numeric(df['Quantity'])
                return df
            return pd.DataFrame(columns=['Symbol', 'Quantity'])
    except Exception as e:
        log.error(f"Error fetching portfolio holdings: {e}")
        return pd.DataFrame(columns=['Symbol', 'Quantity'])

