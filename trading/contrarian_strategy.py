# /trading/contrarian_strategy.py

import logging
import pandas as pd
import numpy as np
import datetime
import pytz
from datetime import timedelta
from typing import Dict, List, Tuple, Optional

from database import db_utils
from data_collection.technicals import calculate_rsi, calculate_bollinger_bands
import config
from sentiment_analysis.advanced_sentiment import fetch_sentiment_data, fetch_price_data

log = logging.getLogger(__name__)

# Thresholds for contrarian strategy
DEFAULT_EXTREME_BEARISH_THRESHOLD = -0.65  # Extreme negative sentiment
DEFAULT_EXTREME_BULLISH_THRESHOLD = 0.65   # Extreme positive sentiment
DEFAULT_OVERBOUGHT_RSI = 78                # RSI overbought threshold
DEFAULT_OVERSOLD_RSI = 22                  # RSI oversold threshold
DEFAULT_BOLLINGER_BAND_PERIODS = 20        # For Bollinger Bands calculation
DEFAULT_BOLLINGER_DEVIATION = 2            # Standard deviations for Bollinger Bands

class ContrarianSignalDetector:
    """
    Detects potential contrarian trading signals based on extreme sentiment
    and technical conditions.
    """
    
    def __init__(self,
                extreme_bullish_threshold: float = DEFAULT_EXTREME_BULLISH_THRESHOLD,
                extreme_bearish_threshold: float = DEFAULT_EXTREME_BEARISH_THRESHOLD,
                overbought_rsi: float = DEFAULT_OVERBOUGHT_RSI,
                oversold_rsi: float = DEFAULT_OVERSOLD_RSI):
        """
        Initialize the contrarian signal detector with threshold parameters.
        
        Args:
            extreme_bullish_threshold: Threshold for extremely bullish sentiment (0.0 to 1.0)
            extreme_bearish_threshold: Threshold for extremely bearish sentiment (-1.0 to 0.0)
            overbought_rsi: RSI threshold for overbought condition
            oversold_rsi: RSI threshold for oversold condition
        """
        self.extreme_bullish_threshold = extreme_bullish_threshold
        self.extreme_bearish_threshold = extreme_bearish_threshold
        self.overbought_rsi = overbought_rsi
        self.oversold_rsi = oversold_rsi
        
    async def check_contrarian_entry(self,
                                   symbol: str,
                                   sentiment_score: float,
                                   price_data: pd.DataFrame) -> Tuple[bool, Optional[str], float]:
        """
        Identifies potential contrarian entry points based on extreme sentiment
        and technical indicators.
        
        Args:
            symbol: Trading pair symbol
            sentiment_score: Current sentiment score
            price_data: DataFrame with price data including open, high, low, close
            
        Returns:
            Tuple of (signal_detected, signal_type, signal_strength)
            where signal_type is 'CONTRARIAN_BUY' or 'CONTRARIAN_SELL'
        """
        # Calculate recent RSI
        if len(price_data) < 14:  # Need at least 14 periods for RSI
            return False, None, 0.0
        
        # Calculate 14-period RSI
        prices = price_data['close'].values
        current_rsi = calculate_rsi(prices, period=14)[-1]
        
        # Calculate Bollinger Bands to detect extreme price levels
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            prices, 
            period=DEFAULT_BOLLINGER_BAND_PERIODS,
            std_dev=DEFAULT_BOLLINGER_DEVIATION
        )
        
        # Get the most recent price and Bollinger Bands values
        current_price = prices[-1]
        bb_upper_last = bb_upper[-1]
        bb_lower_last = bb_lower[-1]
        
        # Calculate price position relative to Bollinger Bands (0.0 = at lower band, 1.0 = at upper band)
        if bb_upper_last > bb_lower_last:  # Avoid division by zero
            bb_position = (current_price - bb_lower_last) / (bb_upper_last - bb_lower_last)
        else:
            bb_position = 0.5  # Default to middle if bands are flat
        
        # Contrarian buy signal: extremely bearish sentiment with oversold conditions
        contrarian_buy = (
            sentiment_score < self.extreme_bearish_threshold and
            (current_rsi < self.oversold_rsi or bb_position < 0.1)
        )
        
        # Contrarian sell signal: extremely bullish sentiment with overbought conditions
        contrarian_sell = (
            sentiment_score > self.extreme_bullish_threshold and
            (current_rsi > self.overbought_rsi or bb_position > 0.9)
        )
        
        # Calculate signal strength: how extreme the conditions are (range 0.0 to 1.0)
        signal_strength = 0.0
        signal_type = None
        
        if contrarian_buy:
            # Combine sentiment and RSI extremeness for buy signal strength
            sentiment_factor = abs(sentiment_score - self.extreme_bearish_threshold) / abs(self.extreme_bearish_threshold)
            rsi_factor = abs(current_rsi - self.oversold_rsi) / self.oversold_rsi
            bb_factor = 1.0 - bb_position
            
            # Weighted average of factors
            signal_strength = 0.4 * sentiment_factor + 0.3 * rsi_factor + 0.3 * bb_factor
            signal_type = "CONTRARIAN_BUY"
            
            log.info(f"Contrarian BUY signal for {symbol}: " +
                    f"Sentiment: {sentiment_score:.4f}, RSI: {current_rsi:.2f}, " +
                    f"BB position: {bb_position:.2f}, Strength: {signal_strength:.2f}")
                    
        elif contrarian_sell:
            # Combine sentiment and RSI extremeness for sell signal strength
            sentiment_factor = abs(sentiment_score - self.extreme_bullish_threshold) / abs(self.extreme_bullish_threshold)
            rsi_factor = abs(current_rsi - self.overbought_rsi) / (100 - self.overbought_rsi)
            bb_factor = bb_position
            
            # Weighted average of factors
            signal_strength = 0.4 * sentiment_factor + 0.3 * rsi_factor + 0.3 * bb_factor
            signal_type = "CONTRARIAN_SELL"
            
            log.info(f"Contrarian SELL signal for {symbol}: " +
                    f"Sentiment: {sentiment_score:.4f}, RSI: {current_rsi:.2f}, " +
                    f"BB position: {bb_position:.2f}, Strength: {signal_strength:.2f}")
        
        return (contrarian_buy or contrarian_sell), signal_type, signal_strength
        
    async def get_historical_contrarian_signals(self,
                                              symbol: str,
                                              lookback_days: int = 30) -> pd.DataFrame:
        """
        Analyzes historical data to find past contrarian signals.
        Useful for validation and strategy refinement.
        
        Args:
            symbol: Trading pair symbol
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with historical signals
        """
        engine = db_utils.engine
        if not engine:
            log.error("Database engine not available")
            return pd.DataFrame()
        
        end_time = datetime.datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=lookback_days)
        
        # Fetch sentiment and price data
        sentiment_data = await fetch_sentiment_data(engine, symbol, start_time, end_time)
        price_data = await fetch_price_data(engine, symbol, config.CANDLE_INTERVAL, start_time, end_time)
        
        if sentiment_data.empty or price_data.empty:
            log.warning(f"Insufficient data for historical signal analysis for {symbol}")
            return pd.DataFrame()
            
        # Resample sentiment data to match price data frequency
        price_freq = pd.infer_freq(price_data.index)
        if price_freq:
            sentiment_resampled = sentiment_data.resample(price_freq).mean().fillna(method='ffill')
        else:
            # If frequency can't be inferred, use the original data
            sentiment_resampled = sentiment_data
            
        # Calculate RSI for the entire price series
        prices = price_data['close'].values
        rsi_values = calculate_rsi(prices, period=14)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            prices, 
            period=DEFAULT_BOLLINGER_BAND_PERIODS,
            std_dev=DEFAULT_BOLLINGER_DEVIATION
        )
        
        # Create DataFrame with all signals
        signals_df = pd.DataFrame(index=price_data.index)
        signals_df['close'] = price_data['close']
        signals_df['returns'] = price_data['returns']
        signals_df['sentiment'] = np.nan
        signals_df['rsi'] = np.nan
        signals_df['bb_position'] = np.nan
        signals_df['signal'] = None
        signals_df['signal_strength'] = 0.0
        
        # Add sentiment data where available
        common_index = signals_df.index.intersection(sentiment_resampled.index)
        signals_df.loc[common_index, 'sentiment'] = sentiment_resampled.loc[common_index, 'sentiment_score']
        
        # Fill sentiment forward to ensure coverage
        signals_df['sentiment'] = signals_df['sentiment'].fillna(method='ffill')
        
        # Add RSI values
        signals_df['rsi'] = rsi_values
        
        # Calculate BB position
        signals_df['bb_upper'] = bb_upper
        signals_df['bb_lower'] = bb_lower
        signals_df['bb_position'] = (signals_df['close'] - signals_df['bb_lower']) / (signals_df['bb_upper'] - signals_df['bb_lower'])
        
        # Iterate through data to find signals
        for i in range(14, len(signals_df)):  # Skip first 14 rows for RSI calculation
            if pd.isna(signals_df['sentiment'].iloc[i]):
                continue
                
            sentiment = signals_df['sentiment'].iloc[i]
            rsi = signals_df['rsi'].iloc[i]
            bb_pos = signals_df['bb_position'].iloc[i]
            
            # Contrarian buy signal
            if (sentiment < self.extreme_bearish_threshold and 
                (rsi < self.oversold_rsi or bb_pos < 0.1)):
                
                # Calculate signal strength
                sentiment_factor = abs(sentiment - self.extreme_bearish_threshold) / abs(self.extreme_bearish_threshold)
                rsi_factor = abs(rsi - self.oversold_rsi) / self.oversold_rsi
                bb_factor = 1.0 - bb_pos
                
                # Weighted average of factors
                signal_strength = 0.4 * sentiment_factor + 0.3 * rsi_factor + 0.3 * bb_factor
                
                signals_df.loc[signals_df.index[i], 'signal'] = "CONTRARIAN_BUY"
                signals_df.loc[signals_df.index[i], 'signal_strength'] = signal_strength
                
            # Contrarian sell signal
            elif (sentiment > self.extreme_bullish_threshold and 
                 (rsi > self.overbought_rsi or bb_pos > 0.9)):
                
                # Calculate signal strength
                sentiment_factor = abs(sentiment - self.extreme_bullish_threshold) / abs(self.extreme_bullish_threshold)
                rsi_factor = abs(rsi - self.overbought_rsi) / (100 - self.overbought_rsi)
                bb_factor = bb_pos
                
                # Weighted average of factors
                signal_strength = 0.4 * sentiment_factor + 0.3 * rsi_factor + 0.3 * bb_factor
                
                signals_df.loc[signals_df.index[i], 'signal'] = "CONTRARIAN_SELL"
                signals_df.loc[signals_df.index[i], 'signal_strength'] = signal_strength
        
        # Filter for only rows with signals
        signals_df = signals_df[~signals_df['signal'].isna()]
        
        # Add future returns columns for performance analysis
        for period in [1, 6, 12, 24, 48]:
            signals_df[f'future_return_{period}'] = price_data['close'].pct_change(periods=period).shift(-period)
            
        return signals_df

    def evaluate_contrarian_performance(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluates performance of contrarian signals on historical data.
        
        Args:
            signals_df: DataFrame with historical signals from get_historical_contrarian_signals
            
        Returns:
            Dictionary with performance metrics
        """
        if signals_df.empty:
            return {}
            
        # Separate buy and sell signals
        buy_signals = signals_df[signals_df['signal'] == 'CONTRARIAN_BUY']
        sell_signals = signals_df[signals_df['signal'] == 'CONTRARIAN_SELL']
        
        performance = {}
        
        # Analyze buy signals performance
        if not buy_signals.empty:
            for period in [1, 6, 12, 24, 48]:
                col = f'future_return_{period}'
                if col in buy_signals.columns:
                    avg_return = buy_signals[col].mean()
                    win_rate = (buy_signals[col] > 0).mean()
                    performance[f'buy_avg_return_{period}'] = avg_return
                    performance[f'buy_win_rate_{period}'] = win_rate
        
        # Analyze sell signals performance
        if not sell_signals.empty:
            for period in [1, 6, 12, 24, 48]:
                col = f'future_return_{period}'
                if col in sell_signals.columns:
                    # For sell signals, negative returns are wins
                    avg_return = sell_signals[col].mean()
                    win_rate = (sell_signals[col] < 0).mean()
                    performance[f'sell_avg_return_{period}'] = avg_return
                    performance[f'sell_win_rate_{period}'] = win_rate
        
        return performance

# Function to be called from the trader module
async def get_contrarian_adjusted_prediction(
    symbol: str,
    prediction: Tuple[int, float],
    sentiment_score: float,
    price_data: pd.DataFrame
) -> Tuple[int, float]:
    """
    Adjust prediction based on contrarian signals.
    
    Args:
        symbol: Trading pair symbol
        prediction: Tuple of (prediction_class, probability)
        sentiment_score: Current sentiment score
        price_data: Recent price data
        
    Returns:
        Adjusted prediction tuple (class, probability)
    """
    pred_class, prob = prediction
    
    # Create contrarian detector with default thresholds
    detector = ContrarianSignalDetector()
    
    # Check for contrarian signal
    signal_detected, signal_type, signal_strength = await detector.check_contrarian_entry(
        symbol, sentiment_score, price_data
    )
    
    if not signal_detected:
        # No contrarian signal, return original prediction
        return prediction
    
    # Adjust prediction based on contrarian signal
    if signal_type == "CONTRARIAN_BUY":
        # Override to buy with strength-adjusted probability
        adjusted_prob = 0.5 + signal_strength * 0.3
        adjusted_class = 1
    elif signal_type == "CONTRARIAN_SELL":
        # Override to sell with strength-adjusted probability
        adjusted_prob = 0.5 - signal_strength * 0.3
        adjusted_class = 0
    else:
        # No valid signal type
        return prediction
    
    log.info(f"Contrarian strategy adjusted prediction for {symbol}: {pred_class}→{adjusted_class}, " +
             f"{prob:.4f}→{adjusted_prob:.4f} (Signal: {signal_type}, Strength: {signal_strength:.2f})")
    
    return (adjusted_class, adjusted_prob)