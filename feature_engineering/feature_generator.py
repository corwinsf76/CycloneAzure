# /feature_engineering/feature_generator.py

import logging
import pandas as pd
import numpy as np
from sqlalchemy import select, and_, Column, Table
from sqlalchemy.sql import Select
import datetime
import pytz
from typing import List, Dict, Optional, Union, Any, cast, Tuple
from pandas import DataFrame, Series, DatetimeIndex
import sys
import os
from tenacity import retry, stop_after_attempt, wait_exponential

# Add the project root directory to PYTHONPATH dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database.db_utils import (
    engine,
    price_data,
    sentiment_analysis_results,
    cryptopanic_sentiment,
    alphavantage_health,
    coingecko_metrics
)

log = logging.getLogger(__name__)

# --- Constants ---
# Define feature calculation parameters (can also be moved to config if needed)
PRICE_LAG_PERIODS = config.FEATURE_LAG_PERIODS # e.g., 20
ROLLING_WINDOWS = ['1h', '4h', '12h', '24h'] # Pandas offset strings for rolling stats
SENTIMENT_WINDOWS = [config.SENTIMENT_AGG_WINDOW_SHORT, config.SENTIMENT_AGG_WINDOW_LONG] # e.g., ['1h', '24h']
PREDICTION_HORIZON = config.PREDICTION_HORIZON_PERIODS # e.g., 3 periods (15 mins for 5m interval)

# --- Helper Functions ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_price_data(engine, symbol: str, interval: str, start_time_utc: datetime.datetime, end_time_utc: datetime.datetime) -> DataFrame:
    """Fetches price data with retry logic."""
    stmt = select(price_data).where(
        price_data.c.symbol == symbol,
        price_data.c.interval == interval,
        price_data.c.open_time >= start_time_utc,
        price_data.c.open_time <= end_time_utc
    ).order_by(price_data.c.open_time)
    
    return pd.read_sql(stmt, engine, index_col='open_time', parse_dates=['open_time'])

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_sentiment_data(engine, start_time_utc: datetime.datetime, end_time_utc: datetime.datetime) -> DataFrame:
    """Fetches sentiment data with retry logic."""
    stmt = select(
        sentiment_analysis_results.c.analyzed_at,
        sentiment_analysis_results.c.sentiment_score
    ).where(
        sentiment_analysis_results.c.analyzed_at >= start_time_utc,
        sentiment_analysis_results.c.analyzed_at <= end_time_utc
    ).order_by(sentiment_analysis_results.c.analyzed_at)
    
    return pd.read_sql(stmt, engine, index_col='analyzed_at', parse_dates=['analyzed_at'])

def _ensure_datetime_index(df: DataFrame) -> DataFrame:
    """Ensure DataFrame has a timezone-aware datetime index."""
    if not isinstance(df.index, DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    idx = cast(DatetimeIndex, df.index)
    if idx.tz is None:
        df.index = idx.tz_localize(pytz.UTC)
    elif idx.tz != pytz.UTC:
        df.index = idx.tz_convert(pytz.UTC)
    
    return df

def _calculate_rolling_sentiment(sentiment_df: DataFrame, price_timestamps: DatetimeIndex, window: str) -> Series:
    """Calculates rolling average sentiment scores aligned with price timestamps."""
    # Ensure index is datetime
    sentiment_df = _ensure_datetime_index(sentiment_df)
    
    # Resample to 1-minute frequency first
    resampled = sentiment_df.resample(rule='1Min')['sentiment_score'].mean()
    
    # Forward fill missing values within the window
    limit = int(pd.Timedelta(window).total_seconds() / 60)  # Convert window to minutes
    
    # Use type ignores for pandas internal typing issues
    filled = resampled.fillna(method='ffill', limit=limit)  # type: ignore
    
    return cast(Series, filled)

def _calculate_target(price_df: pd.DataFrame, periods: int) -> pd.Series:
    """
    Calculates the target variable: 1 if price increased N periods later, 0 otherwise.

    Args:
        price_df (pd.DataFrame): DataFrame with 'close' prices, indexed by time.
        periods (int): Number of periods ahead to look for price increase.

    Returns:
        pd.Series: Series with target variable (1 or 0), indexed by time.
                   NaN where future price is not available.
    """
    future_close = price_df['close'].shift(-periods)
    target = (future_close > price_df['close']).astype(float) # Use float 1.0 / 0.0, NaN where future unknown
    target.name = f'target_up_{periods}p'
    # Where future_close is NaN (at the end of the series), target will be NaN
    log.debug(f"Calculated target variable for {periods} periods ahead.")
    return target

def _fetch_api_features(engine, symbol: str, start_time_utc: datetime.datetime, end_time_utc: datetime.datetime) -> Dict[str, DataFrame]:
    """Fetches features from CryptoPanic, AlphaVantage, and CoinGecko APIs."""
    api_features: Dict[str, DataFrame] = {}
    
    # Fetch CryptoPanic sentiment data
    stmt = select(cryptopanic_sentiment).where(
        and_(
            cryptopanic_sentiment.c.symbol == symbol,
            cryptopanic_sentiment.c.timestamp.between(start_time_utc, end_time_utc)
        )
    ).order_by(cryptopanic_sentiment.c.timestamp)
    
    api_features['cryptopanic'] = pd.read_sql(stmt, engine, index_col='timestamp')

    # Fetch AlphaVantage health data
    stmt = select(alphavantage_health).where(
        and_(
            alphavantage_health.c.symbol == symbol,
            alphavantage_health.c.timestamp.between(start_time_utc, end_time_utc)
        )
    ).order_by(alphavantage_health.c.timestamp)
    
    api_features['alphavantage'] = pd.read_sql(stmt, engine, index_col='timestamp')

    # Fetch CoinGecko metrics data
    stmt = select(coingecko_metrics).where(
        and_(
            coingecko_metrics.c.symbol == symbol,
            coingecko_metrics.c.timestamp.between(start_time_utc, end_time_utc)
        )
    ).order_by(coingecko_metrics.c.timestamp)
    
    api_features['coingecko'] = pd.read_sql(stmt, engine, index_col='timestamp')

    return api_features

def _calculate_api_features(api_data: Dict[str, pd.DataFrame], features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features from API data and adds them to the feature DataFrame."""
    
    # Process CryptoPanic features
    if not api_data['cryptopanic'].empty:
        features_df['cp_sentiment_score'] = api_data['cryptopanic']['sentiment_score']
        features_df['cp_bullish_ratio'] = api_data['cryptopanic']['bullish_count'] / api_data['cryptopanic']['total_articles']
        
        # Rolling features
        for window in ['1h', '4h', '12h', '24h']:
            features_df[f'cp_sent_avg_{window}'] = api_data['cryptopanic']['sentiment_score'].rolling(window, closed='left').mean()
            features_df[f'cp_bull_ratio_{window}'] = (api_data['cryptopanic']['bullish_count'] / api_data['cryptopanic']['total_articles']).rolling(window, closed='left').mean()

    # Process AlphaVantage features
    if not api_data['alphavantage'].empty:
        features_df['av_health_score'] = api_data['alphavantage']['health_score']
        features_df['av_rsi'] = api_data['alphavantage']['rsi']
        features_df['av_macd'] = api_data['alphavantage']['macd']
        features_df['av_macd_signal'] = api_data['alphavantage']['macd_signal']
        
        # Calculate MACD crossover signal
        features_df['av_macd_cross'] = np.where(
            api_data['alphavantage']['macd'] > api_data['alphavantage']['macd_signal'], 1, -1
        )

    # Process CoinGecko features
    if not api_data['coingecko'].empty:
        features_df['cg_market_cap'] = api_data['coingecko']['market_cap']
        features_df['cg_volume'] = api_data['coingecko']['total_volume']
        features_df['cg_price_change_24h'] = api_data['coingecko']['price_change_24h']
        features_df['cg_price_change_7d'] = api_data['coingecko']['price_change_7d']
        features_df['cg_market_rank'] = api_data['coingecko']['market_cap_rank']
        features_df['cg_community_score'] = api_data['coingecko']['community_score']
        features_df['cg_interest_score'] = api_data['coingecko']['public_interest_score']
        
        # Calculate market cap to volume ratio
        features_df['cg_mcap_vol_ratio'] = api_data['coingecko']['market_cap'] / api_data['coingecko']['total_volume']

    return features_df

def generate_technical_features(df: pd.DataFrame, 
                              sma_periods: List[int] = [20, 50], 
                              ema_periods: List[int] = [12, 26],
                              rsi_period: int = 14) -> pd.DataFrame:
    """Generate technical analysis features from price data."""
    df = df.copy()
    
    # SMA features
    for period in sma_periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff()
    
    # EMA features
    for period in ema_periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff()
    
    # RSI
    delta = df['close'].astype(float).diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def generate_volume_features(df: pd.DataFrame, 
                           ma_periods: List[int] = [20, 50]) -> pd.DataFrame:
    """Generate volume-based features."""
    df = df.copy()
    
    # Volume moving averages
    for period in ma_periods:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_sma_{period}_slope'] = df[f'volume_sma_{period}'].diff()
    
    # Volume relative to moving average
    df['volume_sma_ratio'] = df['volume'] / df['volume_sma_20']
    
    return df

def generate_target_variables(df: pd.DataFrame, 
                            horizons: List[int] = [12, 24, 48],  # 1h, 2h, 4h with 5min candles
                            threshold: float = 0.001) -> pd.DataFrame:
    """Generate target variables for different prediction horizons."""
    df = df.copy()
    
    for horizon in horizons:
        # Calculate future returns
        future_price = df['close'].shift(-horizon)
        returns = (future_price - df['close']) / df['close']
        
        # Create binary target (1 if return > threshold)
        df[f'target_up_{horizon}p'] = (returns > threshold).astype(int)
        
    return df

def create_feature_matrix(price_data: pd.DataFrame,
                         sentiment_data: Optional[pd.DataFrame] = None,
                         social_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Create complete feature matrix combining all data sources."""
    df = price_data.copy()
    
    # Technical features
    df = generate_technical_features(df)
    df = generate_volume_features(df)
    
    # Add sentiment features if available
    if sentiment_data is not None:
        df = pd.merge(df, sentiment_data, on='timestamp', how='left')
    
    # Add social features if available
    if social_data is not None:
        df = pd.merge(df, social_data, on='timestamp', how='left')
    
    # Generate target variables
    df = generate_target_variables(df)
    
    # List of feature columns (excluding target variables and timestamp)
    feature_cols = [col for col in df.columns 
                   if not col.startswith('target_') 
                   and col not in ['timestamp', 'symbol']]
    
    return df, feature_cols

# --- Main Feature Generation Function ---

def generate_features_for_symbol(
    symbol: str,
    end_time_utc: datetime.datetime,
    history_duration: Optional[datetime.timedelta] = datetime.timedelta(days=30)
) -> Optional[DataFrame]:
    """Generate features for a symbol up to specified end time."""
    if end_time_utc is None:
        log.error("end_time_utc must not be None")
        return None
        
    if not end_time_utc.tzinfo or end_time_utc.tzinfo.utcoffset(end_time_utc) != datetime.timedelta(0):
        log.error("end_time_utc must be timezone-aware UTC")
        return None

    if history_duration is None:
        history_duration = datetime.timedelta(days=30)
        
    start_time_utc = end_time_utc - history_duration
    
    try:
        # Fetch price data for the specified interval
        interval = config.CANDLE_INTERVAL  # e.g., '5m'
        log.info(f"Fetching price data for {symbol} from {start_time_utc} to {end_time_utc}")
        price_df = fetch_price_data(engine, symbol, interval, start_time_utc, end_time_utc)
        
        if price_df.empty:
            log.warning(f"No price data available for {symbol} in the specified time range")
            return None
            
        # Ensure price data has enough history
        if len(price_df) < PRICE_LAG_PERIODS:
            log.warning(f"Insufficient price history for {symbol}: {len(price_df)} rows < {PRICE_LAG_PERIODS} required")
            return None
            
        # Calculate the target variable
        price_df[f'target_up_{PREDICTION_HORIZON}p'] = _calculate_target(price_df, PREDICTION_HORIZON)
        
        # Generate technical features
        log.debug(f"Generating technical features for {symbol}")
        features_df = generate_technical_features(price_df)
        features_df = generate_volume_features(features_df)
        
        # Fetch API data
        log.debug(f"Fetching API data for {symbol}")
        api_data = _fetch_api_features(engine, symbol.replace('USDT', ''), start_time_utc, end_time_utc)
        
        # Add API features
        log.debug(f"Calculating API features for {symbol}")
        features_df = _calculate_api_features(api_data, features_df)
        
        # Handle missing values
        log.debug(f"Handling missing values")
        # Forward fill and then backward fill remaining NaNs from technical indicators
        # Using dict instead of method='ffill' to avoid Pylance type errors
        features_df = features_df.fillna(method='ffill')  # type: ignore
        features_df = features_df.fillna(method='bfill')  # type: ignore
        
        # Drop rows with NaN target (typically the last few rows depending on prediction horizon)
        target_col = f'target_up_{PREDICTION_HORIZON}p'
        features_df = features_df.dropna(subset=[target_col])
        
        if features_df.empty:
            log.warning(f"After dropping NaNs, feature DataFrame is empty for {symbol}")
            return None
            
        # Final check for any remaining NaNs in the feature columns
        if features_df.isnull().values.any():
            log.warning(f"NaN values detected in features for {symbol}")
            # If you want to drop rows with any NaN, uncomment:
            # features_df = features_df.dropna()
            # Or replace NaNs with default values:
            features_df = features_df.fillna(0)
            
        log.info(f"Successfully generated {len(features_df)} rows of features for {symbol}")
        return features_df
        
    except Exception as e:
        log.error(f"Error generating features for {symbol}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    import argparse

    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')

    # Parse command-line arguments for symbol
    parser = argparse.ArgumentParser(description="Feature Generator for a specific symbol.")
    parser.add_argument('--symbol', type=str, required=True, help="The cryptocurrency symbol (e.g., 'DOGEUSDT').")
    args = parser.parse_args()

    symbol = args.symbol  # Use the symbol provided as an argument
    end_time = datetime.datetime.now(pytz.utc)
    history = pd.Timedelta(days=7)

    print(f"Generating features for {symbol} up to {end_time} with {history} history...")

    features_df = generate_features_for_symbol(symbol, end_time, history_duration=history)

    if features_df is not None and not features_df.empty:
        print(f"\nSuccessfully generated features DataFrame. Shape: {features_df.shape}")
        print("\nColumns:")
        print(features_df.columns.tolist())
        print("\nFeatures DataFrame (last 5 rows):")
        print(features_df.tail())
        target_col_name = f'target_up_{config.PREDICTION_HORIZON_PERIODS}p'
        print(f"\nTarget variable ('{target_col_name}') distribution:")
        print(features_df[target_col_name].value_counts(dropna=False))
        print("\nInfo:")
        features_df.info()
        if features_df.isnull().values.any():
            print("\nWarning: NaNs detected in final feature DataFrame!")
            print(features_df.isnull().sum())
        else:
            print("\nNo NaNs detected in final feature DataFrame.")
    elif features_df is not None and features_df.empty:
        print(f"\nFeature generation ran but resulted in an empty DataFrame (possibly due to NaNs or insufficient history).")
    else:
        print(f"\nFailed to generate features for {symbol}. Check logs and database connection/data.")

    print("\n--- Test Complete ---")
