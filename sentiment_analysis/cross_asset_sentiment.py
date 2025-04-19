# /sentiment_analysis/cross_asset_sentiment.py

import logging
import pandas as pd
import numpy as np
import datetime
import pytz
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Set
import asyncio

from database import db_utils
from utils.rate_limiter import AsyncRateLimiter
import config
from sentiment_analysis.advanced_sentiment import fetch_sentiment_data, fetch_price_data

log = logging.getLogger(__name__)

# Base cryptocurrencies that influence the market
BASE_INFLUENCE_SYMBOLS = ['BTC', 'ETH']

# Cache for spillover effects
_spillover_cache: Dict[str, Dict[str, Tuple[float, datetime.datetime]]] = {}

async def calculate_sentiment_spillover(
    base_symbol: str,
    target_symbol: str,
    lookback_days: int = 30
) -> float:
    """
    Calculate how sentiment from a major crypto spills over to a target crypto.
    
    Args:
        base_symbol: The base symbol (e.g., 'BTC')
        target_symbol: The target symbol to analyze spillover (e.g., 'SOL')
        lookback_days: Number of days to look back for analysis
        
    Returns:
        Spillover coefficient (correlation between base sentiment and target price changes)
    """
    engine = db_utils.engine
    if not engine:
        log.error("Database engine not available")
        return 0.0
    
    # Clean symbols
    base_symbol = base_symbol.replace('USDT', '')
    target_symbol = target_symbol.replace('USDT', '')
    
    # Skip if base and target are the same
    if base_symbol == target_symbol:
        return 0.0
        
    end_time = datetime.datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=lookback_days)
    
    # Fetch sentiment data for base and price data for target
    base_sentiment = await fetch_sentiment_data(engine, base_symbol, start_time, end_time)
    target_price = await fetch_price_data(engine, f"{target_symbol}USDT", config.CANDLE_INTERVAL, start_time, end_time)
    
    if base_sentiment.empty or target_price.empty:
        log.warning(f"Insufficient data for sentiment spillover analysis: {base_symbol} -> {target_symbol}")
        return 0.0
    
    # Resample sentiment data to match price data frequency
    price_freq = pd.infer_freq(target_price.index)
    if price_freq:
        base_sentiment_resampled = base_sentiment.resample(price_freq).mean()
    else:
        # If frequency can't be inferred, use the original data
        base_sentiment_resampled = base_sentiment
        
    # Ensure both dataframes have the same index range
    common_range = [max(base_sentiment_resampled.index.min(), target_price.index.min()),
                   min(base_sentiment_resampled.index.max(), target_price.index.max())]
    
    # Filter data to common range
    base_sentiment_aligned = base_sentiment_resampled[
        (base_sentiment_resampled.index >= common_range[0]) & 
        (base_sentiment_resampled.index <= common_range[1])
    ]
    target_price_aligned = target_price[
        (target_price.index >= common_range[0]) & 
        (target_price.index <= common_range[1])
    ]
    
    if len(base_sentiment_aligned) < 10 or len(target_price_aligned) < 10:
        log.warning(f"Insufficient aligned data points for {base_symbol} -> {target_symbol}")
        return 0.0
    
    # Create a DataFrame with both series
    analysis_df = pd.DataFrame({
        'base_sentiment': base_sentiment_aligned['sentiment_score'],
        'target_returns': target_price_aligned['returns']
    })
    
    # Calculate lagged correlations for different lag periods
    lag_periods = [1, 2, 3, 6, 12, 24]
    correlations = {}
    
    for lag in lag_periods:
        # Shift base sentiment forward to align with future target returns
        analysis_df[f'lagged_sentiment_{lag}'] = analysis_df['base_sentiment'].shift(lag)
        
        # Calculate correlation between lagged sentiment and returns
        corr = analysis_df[f'lagged_sentiment_{lag}'].corr(analysis_df['target_returns'])
        if not np.isnan(corr):
            correlations[lag] = corr
    
    if not correlations:
        return 0.0
    
    # Find the lag with the highest absolute correlation
    optimal_lag, max_corr = max(correlations.items(), key=lambda x: abs(x[1]))
    
    log.info(f"Sentiment spillover from {base_symbol} to {target_symbol}: {max_corr:.4f} (lag: {optimal_lag})")
    return max_corr

async def get_cached_spillover_effect(
    base_symbol: str,
    target_symbol: str,
    max_cache_age_hours: int = 24
) -> float:
    """
    Get cached spillover effect or calculate it if not cached.
    
    Args:
        base_symbol: The base symbol (e.g., 'BTC')
        target_symbol: The target symbol (e.g., 'SOL')
        max_cache_age_hours: Maximum age of cached values in hours
        
    Returns:
        Spillover coefficient
    """
    # Clean symbols
    base_symbol = base_symbol.replace('USDT', '')
    target_symbol = target_symbol.replace('USDT', '')
    
    now = datetime.datetime.now(pytz.UTC)
    
    # Initialize cache for base symbol if needed
    if base_symbol not in _spillover_cache:
        _spillover_cache[base_symbol] = {}
    
    # Check if we have a valid cached value
    if target_symbol in _spillover_cache[base_symbol]:
        spillover, timestamp = _spillover_cache[base_symbol][target_symbol]
        age = now - timestamp
        
        if age.total_seconds() < max_cache_age_hours * 3600:
            log.debug(f"Using cached spillover effect {base_symbol}->{target_symbol}: {spillover:.4f}")
            return spillover
    
    # Calculate new spillover effect
    spillover = await calculate_sentiment_spillover(base_symbol, target_symbol)
    
    # Cache the result
    _spillover_cache[base_symbol][target_symbol] = (spillover, now)
    
    return spillover

async def analyze_market_influence(
    target_symbol: str,
    base_symbols: List[str] = BASE_INFLUENCE_SYMBOLS
) -> Dict[str, float]:
    """
    Analyze how major cryptocurrencies influence the target cryptocurrency.
    
    Args:
        target_symbol: The target symbol to analyze
        base_symbols: List of base symbols that may influence the target
        
    Returns:
        Dictionary mapping base symbols to influence scores
    """
    influence_scores = {}
    
    for base in base_symbols:
        if base in target_symbol:
            # Skip if base is part of the target (e.g., BTCDOWN)
            continue
            
        spillover = await get_cached_spillover_effect(base, target_symbol)
        if abs(spillover) >= 0.1:  # Only consider meaningful correlations
            influence_scores[base] = spillover
    
    return influence_scores

async def get_cross_asset_influence_score(
    target_symbol: str,
    current_base_sentiments: Dict[str, float]
) -> float:
    """
    Calculate the cross-asset influence score for a target asset,
    based on current sentiment of influential base assets.
    
    Args:
        target_symbol: The target symbol to analyze
        current_base_sentiments: Current sentiment scores for base assets
        
    Returns:
        Influence score for the target asset
    """
    target_clean = target_symbol.replace('USDT', '')
    
    # If target is a base asset itself, return its own sentiment
    if target_clean in current_base_sentiments:
        return current_base_sentiments[target_clean]
    
    influence_scores = await analyze_market_influence(target_clean)
    
    if not influence_scores:
        return 0.0
    
    # Calculate weighted average of influential base sentiments
    weighted_score = 0.0
    total_weight = 0.0
    
    for base, influence in influence_scores.items():
        if base in current_base_sentiments:
            # Use absolute influence as weight, but keep the sign for direction
            weight = abs(influence)
            weighted_score += current_base_sentiments[base] * influence
            total_weight += weight
    
    if total_weight > 0:
        return weighted_score / total_weight
    else:
        return 0.0

# Function to be called from the trader module
async def get_cross_asset_adjusted_prediction(
    symbol: str,
    prediction: Tuple[int, float],
    current_market_sentiments: Dict[str, float]
) -> Tuple[int, float]:
    """
    Adjust prediction based on cross-asset sentiment influence.
    
    Args:
        symbol: Trading pair symbol
        prediction: Tuple of (prediction_class, probability)
        current_market_sentiments: Current sentiment scores for major assets
        
    Returns:
        Adjusted prediction tuple (class, probability)
    """
    pred_class, prob = prediction
    
    # Skip adjustment if sentiments not available
    if not current_market_sentiments:
        return prediction
    
    # Get influence score from cross-asset analysis
    influence_score = await get_cross_asset_influence_score(symbol, current_market_sentiments)
    
    if abs(influence_score) < 0.1:
        # Not enough influence to adjust prediction
        return prediction
    
    # Scale the adjustment based on the strength of the influence
    adjustment_factor = influence_score * 0.15
    adjusted_prob = prob + adjustment_factor
    
    # Ensure the probability stays in [0, 1] range
    adjusted_prob = max(0.0, min(1.0, adjusted_prob))
    
    # Adjust prediction class if probability crosses the threshold
    adjusted_class = 1 if adjusted_prob >= 0.5 else 0
    
    log.info(f"Cross-asset adjusted prediction for {symbol}: {pred_class}→{adjusted_class}, " +
             f"{prob:.4f}→{adjusted_prob:.4f} (Influence score: {influence_score:.4f})")
    
    return (adjusted_class, adjusted_prob)

async def get_current_base_sentiments() -> Dict[str, float]:
    """
    Get current sentiment scores for base influential cryptocurrencies.
    
    Returns:
        Dictionary mapping symbols to sentiment scores
    """
    engine = db_utils.engine
    if not engine:
        log.error("Database engine not available")
        return {}
    
    end_time = datetime.datetime.now(pytz.UTC)
    start_time = end_time - timedelta(hours=24)
    
    sentiments = {}
    async with AsyncRateLimiter(rate_limit=5, period=1):  # Limit to 5 queries per second
        for base in BASE_INFLUENCE_SYMBOLS:
            sentiment_df = await fetch_sentiment_data(engine, base, start_time, end_time)
            if not sentiment_df.empty:
                # Calculate weighted average of recent sentiment, giving more weight to recent data
                weights = np.linspace(0.5, 1.0, len(sentiment_df))
                avg_sentiment = np.average(sentiment_df['sentiment_score'], weights=weights)
                sentiments[base] = avg_sentiment
    
    return sentiments