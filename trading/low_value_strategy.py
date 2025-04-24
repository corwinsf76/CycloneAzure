# /trading/low_value_strategy.py

"""
Low-Value Coin Trading Strategy

This module implements specialized trading strategies for cryptocurrencies valued under $1.
It leverages enhanced sentiment analysis, social media metrics, and cross-coin comparisons
to identify trading opportunities in low-value coins which often behave differently than
higher-priced assets.
"""

import logging
import pandas as pd
import numpy as np
import datetime
import pytz
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from sqlalchemy import select, and_

import config
from database.db_utils import async_fetch, engine
from data_collection.binance_client import get_latest_price
from sentiment_analysis.advanced_sentiment import analyze_low_value_coin_sentiment
from sentiment_analysis.analyzer import get_current_sentiment_score

log = logging.getLogger(__name__)

class LowValueCoinStrategy:
    """
    Strategy for trading cryptocurrencies valued under $1, which often have different
    sentiment patterns and volatility characteristics compared to higher-priced coins.
    
    This strategy focuses on:
    1. Social media sentiment and volume spikes
    2. Cross-coin sentiment comparisons
    3. Identifying potential momentum opportunities
    """
    
    def __init__(self):
        """Initialize the low-value coin strategy."""
        self.price_threshold = config.LOW_VALUE_PRICE_THRESHOLD
        self.sentiment_threshold = config.LOW_VALUE_SENTIMENT_THRESHOLD
        self.tweet_volume_factor = config.LOW_VALUE_TWEET_VOLUME_FACTOR
        self.position_percentage = config.LOW_VALUE_POSITION_PERCENTAGE
    
    async def is_low_value_coin(self, symbol: str) -> bool:
        """
        Check if a coin is considered low-value (under the configured price threshold).
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Boolean indicating if this is a low-value coin
        """
        current_price = await get_latest_price(symbol)
        if current_price is None:
            return False
        
        return current_price < self.price_threshold
    
    async def get_recent_sentiment_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve recent specialized sentiment data for the given low-value coin.
        
        Args:
            symbol: Trading pair symbol (with USDT suffix)
            
        Returns:
            Dictionary with sentiment metrics or None if not available
        """
        base_symbol = symbol.replace('USDT', '')
        query = f"""
        SELECT * FROM low_value_coin_sentiment 
        WHERE symbol = '{base_symbol}'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        results = await async_fetch(query)
        if not results:
            return None
        
        return results[0]
    
    async def get_recent_cross_coin_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve recent cross-coin comparison metrics.
        
        Returns:
            Dictionary with cross-coin metrics or None if not available
        """
        query = """
        SELECT * FROM low_value_cross_coin_metrics
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        results = await async_fetch(query)
        if not results:
            return None
        
        return results[0]
    
    async def check_tweet_volume_momentum(self, symbol: str) -> Tuple[bool, float]:
        """
        Check if the coin has abnormally high social media activity,
        which often precedes price movements in low-value coins.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (has_momentum, volume_factor)
        """
        base_symbol = symbol.replace('USDT', '')
        
        # Get current tweet count
        current_data = await self.get_recent_sentiment_data(symbol)
        if not current_data or 'tweet_count' not in current_data:
            return False, 0.0
        
        current_count = current_data['tweet_count']
        
        # Get average tweet count over past week
        query = f"""
        SELECT AVG(tweet_count) as avg_count
        FROM low_value_coin_sentiment 
        WHERE symbol = '{base_symbol}'
        AND timestamp > NOW() - INTERVAL '7 days'
        """
        
        results = await async_fetch(query)
        if not results or not results[0]['avg_count']:
            return False, 0.0
        
        avg_count = results[0]['avg_count']
        
        # Calculate volume factor
        volume_factor = current_count / avg_count if avg_count > 0 else 1.0
        
        # Check if volume meets threshold
        has_momentum = volume_factor >= self.tweet_volume_factor
        
        if has_momentum:
            log.info(f"Detected high tweet volume for {symbol}: {current_count} vs avg {avg_count:.1f} (factor: {volume_factor:.2f})")
        
        return has_momentum, volume_factor
    
    async def check_sentiment_momentum(self, symbol: str) -> Tuple[bool, float]:
        """
        Check if the coin has strong sentiment momentum by comparing
        recent sentiment to historical values.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (has_momentum, sentiment_change)
        """
        base_symbol = symbol.replace('USDT', '')
        
        # Get current sentiment
        current_data = await self.get_recent_sentiment_data(symbol)
        if not current_data or 'avg_sentiment' not in current_data:
            return False, 0.0
        
        current_sentiment = current_data['avg_sentiment']
        
        # Get sentiment from 24 hours ago
        query = f"""
        SELECT avg_sentiment
        FROM low_value_coin_sentiment 
        WHERE symbol = '{base_symbol}'
        AND timestamp < NOW() - INTERVAL '24 hours'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        results = await async_fetch(query)
        if not results:
            return False, 0.0
        
        past_sentiment = results[0]['avg_sentiment']
        
        # Calculate sentiment change
        sentiment_change = current_sentiment - past_sentiment
        
        # Check if sentiment change meets threshold
        has_momentum = sentiment_change > 0.2
        
        if has_momentum:
            log.info(f"Detected positive sentiment momentum for {symbol}: {current_sentiment:.2f} vs past {past_sentiment:.2f} (change: {sentiment_change:.2f})")
        
        return has_momentum, sentiment_change
    
    async def is_most_positive_coin(self, symbol: str) -> bool:
        """
        Check if this coin is currently the most positive among low-value coins.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Boolean indicating if this is the most positive coin
        """
        base_symbol = symbol.replace('USDT', '')
        
        cross_metrics = await self.get_recent_cross_coin_metrics()
        if not cross_metrics or 'most_positive_coin' not in cross_metrics:
            return False
        
        return cross_metrics['most_positive_coin'] == base_symbol
    
    async def analyze_trading_opportunity(self, symbol: str) -> Tuple[int, float, Dict[str, Any]]:
        """
        Analyze if a low-value coin presents a trading opportunity.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (trade_decision, confidence, metadata)
            where trade_decision is 1 for buy, 0 for hold/sell
        """
        metadata = {}
        
        # Check if this is a low-value coin
        is_low_value = await self.is_low_value_coin(symbol)
        if not is_low_value:
            return 0, 0.0, {'reason': 'not_low_value_coin'}
        
        # Get sentiment data
        sentiment_data = await self.get_recent_sentiment_data(symbol)
        if not sentiment_data:
            # Fallback to standard sentiment
            standard_sentiment = await get_current_sentiment_score(symbol)
            sentiment_data = {'avg_sentiment': standard_sentiment}
        
        metadata['sentiment_data'] = sentiment_data
        
        # Check if sentiment meets minimum threshold
        avg_sentiment = sentiment_data.get('avg_sentiment', 0)
        if avg_sentiment < self.sentiment_threshold:
            return 0, 0.0, {**metadata, 'reason': 'sentiment_below_threshold'}
        
        # Factors that contribute to the trading decision
        factors = []
        
        # Factor 1: Check tweet volume momentum
        has_volume_momentum, volume_factor = await self.check_tweet_volume_momentum(symbol)
        metadata['volume_factor'] = volume_factor
        if has_volume_momentum:
            factors.append(('volume_momentum', min(0.3, 0.1 * (volume_factor / self.tweet_volume_factor))))
        
        # Factor 2: Check sentiment momentum
        has_sentiment_momentum, sentiment_change = await self.check_sentiment_momentum(symbol)
        metadata['sentiment_change'] = sentiment_change
        if has_sentiment_momentum:
            factors.append(('sentiment_momentum', min(0.3, sentiment_change)))
        
        # Factor 3: Check if most positive among low-value coins
        is_top_sentiment = await self.is_most_positive_coin(symbol)
        metadata['is_most_positive'] = is_top_sentiment
        if is_top_sentiment:
            factors.append(('top_sentiment', 0.2))
        
        # Factor 4: Check FinBERT specialized scores if available
        if 'avg_positive' in sentiment_data and 'avg_negative' in sentiment_data:
            polarity = sentiment_data['avg_positive'] - sentiment_data['avg_negative']
            if polarity > 0.2:
                factors.append(('finbert_polarity', min(0.2, polarity)))
                metadata['finbert_polarity'] = polarity
        
        # Calculate overall confidence based on factors
        if not factors:
            return 0, 0.0, {**metadata, 'reason': 'no_positive_factors'}
        
        # Calculate confidence as weighted sum of factors
        confidence = sum(weight for _, weight in factors)
        metadata['factors'] = factors
        
        # Minimum confidence threshold
        if confidence < 0.3:
            return 0, confidence, {**metadata, 'reason': 'confidence_too_low'}
        
        # Return buy decision with confidence score
        trade_decision = 1  # Buy
        metadata['decision_reason'] = 'low_value_strategy'
        
        log.info(f"Low-value coin strategy recommends BUY for {symbol} with confidence {confidence:.2f}")
        log.debug(f"Decision factors: {factors}")
        
        return trade_decision, confidence, metadata

async def get_low_value_adjusted_prediction(
    symbol: str,
    original_prediction: Tuple[int, float]
) -> Tuple[int, float]:
    """
    Adjust a prediction using the low-value coin strategy.
    
    Args:
        symbol: Trading pair symbol
        original_prediction: Tuple of (prediction_class, confidence)
        
    Returns:
        Adjusted prediction tuple
    """
    if not config.LOW_VALUE_COIN_ENABLED:
        return original_prediction
    
    try:
        strategy = LowValueCoinStrategy()
        is_low_value = await strategy.is_low_value_coin(symbol)
        
        if not is_low_value:
            # Not a low-value coin, return original prediction
            return original_prediction
        
        # Get low-value strategy recommendation
        decision, confidence, metadata = await strategy.analyze_trading_opportunity(symbol)
        
        # If low-value strategy gave a buy signal with higher confidence, use it
        original_class, original_confidence = original_prediction
        
        # If original signal matches low-value signal, boost confidence
        if decision == original_class and decision == 1:  # Both recommend buy
            combined_confidence = min(0.9, original_confidence + 0.1)
            log.info(f"Low-value strategy boosted BUY confidence for {symbol}: {original_confidence:.2f} → {combined_confidence:.2f}")
            return (decision, combined_confidence)
        
        # If low-value strategy has higher confidence than original, overrule
        if confidence > original_confidence and decision == 1:
            log.info(f"Low-value strategy overruled original prediction for {symbol}: {original_class} → {decision}, {original_confidence:.2f} → {confidence:.2f}")
            return (decision, confidence)
        
        # Otherwise keep original prediction
        return original_prediction
        
    except Exception as e:
        log.error(f"Error applying low-value coin strategy: {e}")
        return original_prediction