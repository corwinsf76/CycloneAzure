# /sentiment_analysis/advanced_sentiment.py

import logging
import pandas as pd
import numpy as np
import datetime
import pytz
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
import asyncio

from database import db_utils
from sentiment_analysis.analyzer import get_current_sentiment_score
import config

log = logging.getLogger(__name__)

# Default lag settings - Will be tuned empirically over time
DEFAULT_LAG_WINDOWS = [1, 3, 7, 14, 30]  # Days
DEFAULT_LAG_WEIGHTS = [0.40, 0.25, 0.15, 0.10, 0.10]  # Higher weights for recent periods


async def fetch_sentiment_data(
    engine,
    symbol: str, 
    start_time: datetime.datetime, 
    end_time: datetime.datetime
) -> pd.DataFrame:
    """
    Fetch historical sentiment data for a symbol within a date range.
    
    Args:
        engine: SQLAlchemy engine
        symbol: Trading pair symbol
        start_time: Start time for data fetch
        end_time: End time for data fetch
        
    Returns:
        DataFrame with sentiment data
    """
    if not engine:
        log.error("Database engine not available")
        return pd.DataFrame()
    
    # Fetch social sentiment from database
    try:
        stmt = f"""
        SELECT timestamp, 
               sentiment_score,
               source
        FROM sentiment_data
        WHERE symbol = '{symbol}'
          AND timestamp BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY timestamp
        """
        
        sentiment_df = pd.read_sql(stmt, engine)
        
        if sentiment_df.empty:
            log.warning(f"No sentiment data found for {symbol}")
            return pd.DataFrame()
            
        # Set timestamp as index and ensure it's timezone-aware
        sentiment_df.set_index('timestamp', inplace=True)
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
        if sentiment_df.index.tzinfo is None:
            sentiment_df.index = sentiment_df.index.tz_localize(pytz.UTC)
            
        # Aggregate sentiment by day (average across sources)
        daily_sentiment = sentiment_df.groupby(pd.Grouper(freq='D')).mean()
        
        # Fill missing days with forward fill then backward fill
        daily_sentiment = daily_sentiment.resample('D').asfreq()
        daily_sentiment = daily_sentiment.fillna(method='ffill').fillna(method='bfill')
        
        return daily_sentiment
        
    except Exception as e:
        log.error(f"Error fetching sentiment data: {e}")
        return pd.DataFrame()


async def fetch_price_data(
    engine,
    symbol: str, 
    interval: str,
    start_time: datetime.datetime, 
    end_time: datetime.datetime
) -> pd.DataFrame:
    """
    Fetch historical price data for a symbol within a date range.
    
    Args:
        engine: SQLAlchemy engine
        symbol: Trading pair symbol
        interval: Candle interval (e.g., '1h', '1d')
        start_time: Start time for data fetch
        end_time: End time for data fetch
        
    Returns:
        DataFrame with price data
    """
    if not engine:
        log.error("Database engine not available")
        return pd.DataFrame()
    
    # Fetch price data from database
    try:
        stmt = f"""
        SELECT open_time as timestamp, 
               open, high, low, close, volume
        FROM price_data
        WHERE symbol = '{symbol}'
          AND interval = '{interval}'
          AND open_time BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY open_time
        """
        
        price_df = pd.read_sql(stmt, engine)
        
        if price_df.empty:
            log.warning(f"No price data found for {symbol}")
            return pd.DataFrame()
            
        # Set timestamp as index and ensure it's timezone-aware
        price_df.set_index('timestamp', inplace=True)
        price_df.index = pd.to_datetime(price_df.index)
        if price_df.index.tzinfo is None:
            price_df.index = price_df.index.tz_localize(pytz.UTC)
            
        # Calculate returns
        price_df['returns'] = price_df['close'].pct_change()
        
        if interval == '1h':
            # For hourly data, create daily aggregates
            daily_price = price_df.groupby(pd.Grouper(freq='D')).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'returns': 'sum'
            })
            return daily_price
        else:
            return price_df
            
    except Exception as e:
        log.error(f"Error fetching price data: {e}")
        return pd.DataFrame()


async def analyze_sentiment_lag_effect(
    symbol: str,
    lookback_days: int = 90
) -> Dict[str, float]:
    """
    Analyze the lag effect between sentiment changes and price movements.
    
    Args:
        symbol: Trading pair symbol
        lookback_days: Number of days to analyze
        
    Returns:
        Dictionary with lag analysis results
    """
    engine = db_utils.engine
    if not engine:
        log.error("Database engine not available")
        return {}
    
    end_time = datetime.datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=lookback_days)
    
    # Fetch sentiment and price data
    sentiment_data = await fetch_sentiment_data(engine, symbol, start_time, end_time)
    price_data = await fetch_price_data(engine, symbol, config.CANDLE_INTERVAL, start_time, end_time)
    
    if sentiment_data.empty or price_data.empty:
        log.warning(f"Insufficient data for lag analysis for {symbol}")
        return {}
    
    # Align sentiment and price data to ensure they have the same dates
    common_index = sentiment_data.index.intersection(price_data.index)
    if len(common_index) < 10:  # Need at least 10 data points for meaningful analysis
        log.warning(f"Not enough overlapping data points for lag analysis for {symbol}")
        return {}
        
    aligned_sentiment = sentiment_data.loc[common_index]
    aligned_price = price_data.loc[common_index]
    
    # Calculate correlations for different lag periods
    lag_correlations = {}
    for lag in range(1, 15):  # Check lags from 1 to 14 days
        # Shift sentiment back by 'lag' days to align with future price movements
        lagged_sentiment = aligned_sentiment['sentiment_score'].shift(lag)
        
        # Calculate correlation with future returns
        correlation = lagged_sentiment.corr(aligned_price['returns'])
        lag_correlations[lag] = correlation
    
    # Find the lag with the strongest correlation
    if not lag_correlations:
        return {}
        
    abs_correlations = {lag: abs(corr) for lag, corr in lag_correlations.items()}
    strongest_lag = max(abs_correlations, key=abs_correlations.get)
    strongest_corr = lag_correlations[strongest_lag]
    
    # Calculate weighted sentiment using different lag windows
    results = {
        'strongest_lag': strongest_lag,
        'strongest_correlation': strongest_corr,
        'lag_correlations': lag_correlations
    }
    
    log.info(f"Sentiment lag analysis for {symbol}: strongest effect at lag={strongest_lag} days with correlation={strongest_corr:.4f}")
    
    return results


async def get_weighted_historical_sentiment(
    symbol: str,
    lag_windows: List[int] = DEFAULT_LAG_WINDOWS,
    lag_weights: List[float] = DEFAULT_LAG_WEIGHTS
) -> float:
    """
    Calculate a weighted sentiment score using historical sentiment data with different lag windows.
    
    Args:
        symbol: Trading pair symbol
        lag_windows: List of lag window sizes in days
        lag_weights: List of weights corresponding to each lag window
        
    Returns:
        Weighted historical sentiment score
    """
    if len(lag_windows) != len(lag_weights):
        log.error("Lag windows and weights must have the same length")
        return 0.0
        
    engine = db_utils.engine
    if not engine:
        log.error("Database engine not available")
        return 0.0
    
    # Get current time
    end_time = datetime.datetime.now(pytz.UTC)
    # Get data for the longest lag window
    max_lookback = max(lag_windows) + 1
    start_time = end_time - timedelta(days=max_lookback)
    
    # Fetch sentiment data
    sentiment_data = await fetch_sentiment_data(engine, symbol, start_time, end_time)
    
    if sentiment_data.empty:
        log.warning(f"No sentiment data available for {symbol}")
        return 0.0
    
    # Calculate weighted sentiment for each lag window
    weighted_sentiment = 0.0
    total_applied_weight = 0.0
    
    for window, weight in zip(lag_windows, lag_weights):
        # Define the period for this lag window
        window_end = end_time - timedelta(days=1)  # Yesterday
        window_start = window_end - timedelta(days=window)
        
        # Filter sentiment data for this window
        window_filter = (sentiment_data.index >= window_start) & (sentiment_data.index <= window_end)
        window_sentiment = sentiment_data.loc[window_filter]
        
        if not window_sentiment.empty:
            # Calculate average sentiment for this window
            avg_sentiment = window_sentiment['sentiment_score'].mean()
            weighted_sentiment += avg_sentiment * weight
            total_applied_weight += weight
        
    # Normalize by actually applied weights
    if total_applied_weight > 0:
        weighted_sentiment /= total_applied_weight
    
    return weighted_sentiment


async def get_sentiment_adjusted_prediction(
    symbol: str,
    base_prediction: Tuple[int, float],
    current_sentiment: Optional[float] = None
) -> Tuple[int, float]:
    """
    Adjust prediction based on lag-weighted sentiment analysis.
    
    Args:
        symbol: Trading pair symbol
        base_prediction: Tuple of (prediction_class, probability)
        current_sentiment: Current sentiment score, if already available
        
    Returns:
        Adjusted prediction tuple (class, probability)
    """
    pred_class, prob = base_prediction
    
    # Get lag analysis results for this symbol
    lag_results = await analyze_sentiment_lag_effect(symbol)
    
    # If we don't have lag analysis or the correlation is weak, return original prediction
    if not lag_results or abs(lag_results.get('strongest_correlation', 0)) < 0.1:
        log.info(f"No significant sentiment lag effect found for {symbol}, using original prediction")
        return base_prediction
    
    # Get current sentiment if not provided
    if current_sentiment is None:
        current_sentiment = await get_current_sentiment_score(symbol)
    
    # Get weighted historical sentiment
    weighted_sentiment = await get_weighted_historical_sentiment(symbol)
    
    # Calculate an adjustment factor based on current and historical sentiment
    # This is a simple approach that can be refined with more data analysis
    sentiment_delta = current_sentiment - weighted_sentiment
    
    # If sentiment correlation is positive, positive sentiment delta should increase buy probability
    # If sentiment correlation is negative, positive sentiment delta should decrease buy probability
    corr_direction = 1 if lag_results.get('strongest_correlation', 0) > 0 else -1
    
    # Calculate adjustment (range: -0.2 to 0.2)
    adjustment = 0.2 * sentiment_delta * corr_direction
    
    # Apply the adjustment to the probability
    adjusted_prob = max(0.01, min(0.99, prob + adjustment))
    
    # Determine if the prediction class should change
    adjusted_class = 1 if adjusted_prob > 0.5 else 0
    
    # Only log if the adjustment is significant
    if abs(adjusted_prob - prob) > 0.05 or adjusted_class != pred_class:
        log.info(f"Sentiment adjustment for {symbol}: {pred_class}→{adjusted_class}, " +
                f"{prob:.4f}→{adjusted_prob:.4f} (Current: {current_sentiment:.4f}, Weighted: {weighted_sentiment:.4f})")
    
    return (adjusted_class, adjusted_prob)