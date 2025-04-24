"""
Cross-Asset Sentiment Analysis Module - Now with async support

This module analyzes sentiment relationships between different crypto assets
and provides insights into market-wide sentiment patterns.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import asyncio

import config
from database.db_utils import async_df_to_db, async_bulk_insert
from .analyzer import analyze_texts_batch, get_aggregate_sentiment

log = logging.getLogger(__name__)

async def analyze_cross_asset_sentiment(
    symbols: List[str],
    lookback_days: int = 7
) -> Optional[Dict[str, Any]]:
    """
    Analyze sentiment relationships between different assets over time.
    Returns correlation matrix and other cross-asset metrics.
    """
    if not symbols:
        return None
    
    try:
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get sentiment data for each symbol
        sentiments = {}
        for symbol in symbols:
            symbol_sentiment = await get_symbol_sentiment_history(
                symbol,
                start_date,
                end_date
            )
            if symbol_sentiment is not None:
                sentiments[symbol] = symbol_sentiment
        
        if not sentiments:
            return None
        
        # Create DataFrame with sentiment scores
        df = pd.DataFrame(sentiments)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Find highly correlated pairs
        correlated_pairs = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]
                if sym1 in corr_matrix.index and sym2 in corr_matrix.columns:
                    corr = corr_matrix.loc[sym1, sym2]
                    if abs(corr) > 0.7:  # High correlation threshold
                        correlated_pairs.append({
                            'symbol1': sym1,
                            'symbol2': sym2,
                            'correlation': float(corr)
                        })
        
        # Calculate market-wide sentiment metrics
        market_sentiment = {
            'avg_sentiment': float(df.mean().mean()),
            'sentiment_volatility': float(df.std().mean()),
            'sentiment_dispersion': float(df.std(axis=1).mean()),
            'timestamp': datetime.now(pytz.UTC)
        }
        
        # Store results
        results = {
            'correlation_matrix': corr_matrix.to_dict(),
            'correlated_pairs': correlated_pairs,
            'market_sentiment': market_sentiment,
            'timestamp': datetime.now(pytz.UTC)
        }
        
        await async_bulk_insert([results], 'cross_asset_sentiment')
        return results
        
    except Exception as e:
        log.error(f"Error in cross-asset sentiment analysis: {e}")
        return None

async def get_symbol_sentiment_history(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.Series]:
    """
    Get historical sentiment data for a symbol.
    Returns time series of sentiment scores.
    """
    try:
        # Query sentiment data from database
        query = f"""
        SELECT timestamp, sentiment_score
        FROM sentiment_analysis_results
        WHERE symbol = $1 
        AND timestamp BETWEEN $2 AND $3
        ORDER BY timestamp
        """
        
        async with config.DB_POOL.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_date, end_date)
            
        if not rows:
            return None
        
        # Convert to pandas Series
        data = {row['timestamp']: row['sentiment_score'] for row in rows}
        return pd.Series(data)
        
    except Exception as e:
        log.error(f"Error getting sentiment history for {symbol}: {e}")
        return None

async def get_sentiment_divergence(
    symbols: List[str],
    lookback_days: int = 7
) -> Optional[Dict[str, Any]]:
    """
    Analyze divergences between asset sentiment trends.
    Returns pairs of assets with diverging sentiment.
    """
    try:
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get sentiment trends
        trends = {}
        for symbol in symbols:
            history = await get_symbol_sentiment_history(symbol, start_date, end_date)
            if history is not None:
                # Calculate trend using linear regression
                x = np.arange(len(history))
                y = history.values
                trend = np.polyfit(x, y, 1)[0]  # Get slope
                trends[symbol] = trend
        
        if not trends:
            return None
        
        # Find diverging pairs
        divergences = []
        for sym1 in trends:
            for sym2 in trends:
                if sym1 < sym2:  # Avoid duplicates
                    trend1, trend2 = trends[sym1], trends[sym2]
                    if trend1 * trend2 < 0:  # Opposite trends
                        divergences.append({
                            'symbol1': sym1,
                            'symbol2': sym2,
                            'trend1': float(trend1),
                            'trend2': float(trend2)
                        })
        
        results = {
            'divergences': divergences,
            'timestamp': datetime.now(pytz.UTC)
        }
        
        await async_bulk_insert([results], 'sentiment_divergences')
        return results
        
    except Exception as e:
        log.error(f"Error analyzing sentiment divergence: {e}")
        return None

async def get_sentiment_momentum(
    symbols: List[str],
    short_window: int = 1,
    long_window: int = 7
) -> Optional[Dict[str, Any]]:
    """
    Calculate sentiment momentum indicators for assets.
    Returns momentum scores for each asset.
    """
    try:
        end_date = datetime.now(pytz.UTC)
        long_start = end_date - timedelta(days=long_window)
        short_start = end_date - timedelta(days=short_window)
        
        momentum_scores = {}
        for symbol in symbols:
            # Get long and short-term sentiment
            long_hist = await get_symbol_sentiment_history(symbol, long_start, end_date)
            short_hist = await get_symbol_sentiment_history(symbol, short_start, end_date)
            
            if long_hist is not None and short_hist is not None:
                long_avg = float(long_hist.mean())
                short_avg = float(short_hist.mean())
                momentum = short_avg - long_avg
                
                momentum_scores[symbol] = {
                    'short_term_sentiment': short_avg,
                    'long_term_sentiment': long_avg,
                    'momentum': momentum
                }
        
        if not momentum_scores:
            return None
        
        results = {
            'momentum_scores': momentum_scores,
            'timestamp': datetime.now(pytz.UTC)
        }
        
        await async_bulk_insert([results], 'sentiment_momentum')
        return results
        
    except Exception as e:
        log.error(f"Error calculating sentiment momentum: {e}")
        return None