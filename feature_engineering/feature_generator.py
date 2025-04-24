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
    coingecko_metrics,
    low_value_coin_sentiment,
    low_value_cross_coin_metrics,
    news_data,
    social_media_data
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_low_value_coin_sentiment_data(engine, symbol: str, start_time_utc: datetime.datetime, end_time_utc: datetime.datetime) -> DataFrame:
    """
    Fetches specialized sentiment data for low-value coins.
    
    This function retrieves sentiment analysis specifically for cryptocurrencies valued under $1,
    which provides more targeted sentiment metrics for these types of coins.
    
    Args:
        engine: SQLAlchemy engine
        symbol: Cryptocurrency symbol (without USDT suffix)
        start_time_utc: Start time for data retrieval
        end_time_utc: End time for data retrieval
        
    Returns:
        DataFrame with sentiment data for the specified symbol
    """
    stmt = select(low_value_coin_sentiment).where(
        and_(
            low_value_coin_sentiment.c.symbol == symbol,
            low_value_coin_sentiment.c.timestamp.between(start_time_utc, end_time_utc)
        )
    ).order_by(low_value_coin_sentiment.c.timestamp)
    
    df = pd.read_sql(stmt, engine, index_col='timestamp', parse_dates=['timestamp'])
    
    if df.empty:
        log.debug(f"No low-value coin sentiment data available for {symbol}")
    else:
        log.debug(f"Retrieved {len(df)} rows of low-value coin sentiment data for {symbol}")
        
    return df

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_cross_coin_sentiment_metrics(engine, start_time_utc: datetime.datetime, end_time_utc: datetime.datetime) -> DataFrame:
    """
    Fetches cross-coin sentiment metrics that compare sentiment across all low-value coins.
    
    This provides market-wide sentiment context which can be valuable for identifying 
    sentiment-driven trading opportunities in the broader low-value coin market.
    
    Args:
        engine: SQLAlchemy engine
        start_time_utc: Start time for data retrieval
        end_time_utc: End time for data retrieval
        
    Returns:
        DataFrame with cross-coin sentiment metrics
    """
    stmt = select(low_value_cross_coin_metrics).where(
        low_value_cross_coin_metrics.c.timestamp.between(start_time_utc, end_time_utc)
    ).order_by(low_value_cross_coin_metrics.c.timestamp)
    
    df = pd.read_sql(stmt, engine, index_col='timestamp', parse_dates=['timestamp'])
    
    if df.empty:
        log.debug("No cross-coin sentiment metrics available")
    else:
        log.debug(f"Retrieved {len(df)} rows of cross-coin sentiment metrics")
        
    return df

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_general_sentiment_data(engine, symbol: str, start_time_utc: datetime.datetime, end_time_utc: datetime.datetime) -> DataFrame:
    """Fetches sentiment scores from news_data and social_media_data."""
    log.debug(f"Fetching general sentiment for {symbol} from {start_time_utc} to {end_time_utc}")
    
    # Fetch from news_data
    news_stmt = select(
        news_data.c.published_at.label('timestamp'),
        news_data.c.sentiment_score,
        news_data.c.sentiment_magnitude
    ).where(
        and_(
            news_data.c.symbol == symbol,
            news_data.c.published_at.between(start_time_utc, end_time_utc),
            news_data.c.sentiment_score.is_not(None)
        )
    )
    news_df = pd.read_sql(news_stmt, engine, index_col='timestamp', parse_dates=['timestamp'])
    news_df['source'] = 'news'
    
    # Fetch from social_media_data
    social_stmt = select(
        social_media_data.c.created_at.label('timestamp'),
        social_media_data.c.sentiment_score,
        social_media_data.c.sentiment_magnitude,
        social_media_data.c.platform
    ).where(
        and_(
            social_media_data.c.symbol == symbol,
            social_media_data.c.created_at.between(start_time_utc, end_time_utc),
            social_media_data.c.sentiment_score.is_not(None)
        )
    )
    social_df = pd.read_sql(social_stmt, engine, index_col='timestamp', parse_dates=['timestamp'])
    social_df['source'] = social_df['platform']
    social_df = social_df.drop(columns=['platform'])

    # Combine and return
    combined_df = pd.concat([news_df, social_df])
    combined_df = _ensure_datetime_index(combined_df)
    combined_df = combined_df.sort_index()
    
    log.debug(f"Fetched {len(combined_df)} general sentiment records for {symbol}")
    return combined_df

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
    sentiment_df = _ensure_datetime_index(sentiment_df)
    resampled = sentiment_df.resample(rule='1Min')['sentiment_score'].mean()
    limit = int(pd.Timedelta(window).total_seconds() / 60)
    filled = resampled.fillna(method='ffill', limit=limit)
    return cast(Series, filled)

def _calculate_general_sentiment_features(general_sentiment_df: DataFrame, price_index: DatetimeIndex, windows: List[str] = SENTIMENT_WINDOWS) -> DataFrame:
    """Calculates aggregated features from general news/social sentiment."""
    if general_sentiment_df.empty:
        return pd.DataFrame(index=price_index)

    sentiment_features = pd.DataFrame(index=price_index)
    general_sentiment_df = _ensure_datetime_index(general_sentiment_df)

    resample_interval = config.CANDLE_INTERVAL
    resampled_sentiment = general_sentiment_df.resample(resample_interval).agg(
        sentiment_score=('sentiment_score', 'mean'),
        sentiment_magnitude=('sentiment_magnitude', 'mean'),
        record_count=('source', 'count')
    ).reindex(price_index)

    for window in windows:
        rolling_agg = resampled_sentiment.rolling(window, closed='left').agg(
            mean_score=('sentiment_score', 'mean'),
            mean_magnitude=('sentiment_magnitude', 'mean'),
            sum_count=('record_count', 'sum')
        )
        sentiment_features[f'gen_sent_score_avg_{window}'] = rolling_agg['mean_score']
        sentiment_features[f'gen_sent_mag_avg_{window}'] = rolling_agg['mean_magnitude']
        sentiment_features[f'gen_sent_count_{window}'] = rolling_agg['sum_count']

        for source in ['news', 'twitter', 'reddit']:
            source_df = general_sentiment_df[general_sentiment_df['source'] == source]
            if not source_df.empty:
                resampled_source = source_df.resample(resample_interval).agg(
                    sentiment_score=('sentiment_score', 'mean'),
                    record_count=('source', 'count')
                ).reindex(price_index)
                 
                rolling_source_agg = resampled_source.rolling(window, closed='left').agg(
                    mean_score=('sentiment_score', 'mean'),
                    sum_count=('record_count', 'sum')
                )
                sentiment_features[f'{source}_sent_score_avg_{window}'] = rolling_source_agg['mean_score']
                sentiment_features[f'{source}_sent_count_{window}'] = rolling_source_agg['sum_count']

    for window in windows:
        col_name = f'gen_sent_score_avg_{window}'
        if col_name in sentiment_features:
            sentiment_features[f'{col_name}_roc'] = sentiment_features[col_name].pct_change()

    return sentiment_features

def generate_technical_features(df: pd.DataFrame, 
                              sma_periods: List[int] = [20, 50], 
                              ema_periods: List[int] = [12, 26],
                              rsi_period: int = 14) -> pd.DataFrame:
    """Generate technical analysis features from price data."""
    df = df.copy()
    
    for period in sma_periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff()
    
    for period in ema_periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff()
    
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
    
    for period in ma_periods:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_sma_{period}_slope'] = df[f'volume_sma_{period}'].diff()
    
    df['volume_sma_ratio'] = df['volume'] / df['volume_sma_20']
    
    return df

def generate_target_variables(df: pd.DataFrame, 
                            horizons: List[int] = [12, 24, 48], 
                            threshold: float = 0.001) -> pd.DataFrame:
    """Generate target variables for different prediction horizons."""
    df = df.copy()
    
    for horizon in horizons:
        future_price = df['close'].shift(-horizon)
        returns = (future_price - df['close']) / df['close']
        df[f'target_up_{horizon}p'] = (returns > threshold).astype(int)
        
    return df

def create_feature_matrix(price_data: pd.DataFrame,
                         sentiment_data: Optional[pd.DataFrame] = None,
                         social_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Create complete feature matrix combining all data sources."""
    df = price_data.copy()
    
    df = generate_technical_features(df)
    df = generate_volume_features(df)
    
    if sentiment_data is not None:
        df = pd.merge(df, sentiment_data, on='timestamp', how='left')
    
    if social_data is not None:
        df = pd.merge(df, social_data, on='timestamp', how='left')
    
    df = generate_target_variables(df)
    
    feature_cols = [col for col in df.columns 
                   if not col.startswith('target_') 
                   and col not in ['timestamp', 'symbol']]
    
    return df, feature_cols

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
        interval = config.CANDLE_INTERVAL
        log.info(f"Fetching price data for {symbol} from {start_time_utc} to {end_time_utc}")
        price_df = fetch_price_data(engine, symbol, interval, start_time_utc, end_time_utc)
        
        if price_df.empty or len(price_df) < PRICE_LAG_PERIODS:
             log.warning(f"Insufficient price data for {symbol}. Required: {PRICE_LAG_PERIODS}, Found: {len(price_df)}")
             return None
            
        price_df[f'target_up_{PREDICTION_HORIZON}p'] = _calculate_target(price_df, PREDICTION_HORIZON)
        
        log.debug(f"Generating technical features for {symbol}")
        features_df = generate_technical_features(price_df)
        features_df = generate_volume_features(features_df)
        
        features_df['symbol'] = symbol
        
        log.debug(f"Fetching API data for {symbol}")
        base_symbol = symbol.replace('USDT', '')
        api_data = _fetch_api_features(engine, base_symbol, start_time_utc, end_time_utc)
        
        log.debug(f"Calculating API features for {symbol}")
        features_df = _calculate_api_features(api_data, features_df)

        log.debug(f"Fetching general news/social sentiment data for {symbol}")
        general_sentiment_df = _fetch_general_sentiment_data(engine, symbol, start_time_utc, end_time_utc)
        
        log.debug(f"Calculating general sentiment features for {symbol}")
        general_sentiment_features = _calculate_general_sentiment_features(general_sentiment_df, features_df.index)
        
        features_df = pd.merge(features_df, general_sentiment_features, left_index=True, right_index=True, how='left')
        log.info(f"Added {len(general_sentiment_features.columns)} general sentiment features for {symbol}")

        is_low_value = False
        recent_close = price_df['close'].iloc[-1] if not price_df.empty else None
        if recent_close is not None and recent_close < 1.0:
            is_low_value = True
            log.info(f"{symbol} identified as a low-value coin (price: ${recent_close:.4f})")
            
            log.debug(f"Fetching low-value coin sentiment data for {base_symbol}")
            low_value_data = fetch_low_value_coin_sentiment_data(engine, base_symbol, start_time_utc, end_time_utc)
            
            log.debug("Fetching cross-coin sentiment metrics")
            cross_coin_data = fetch_cross_coin_sentiment_metrics(engine, start_time_utc, end_time_utc)
            
            log.debug("Calculating low-value coin sentiment features")
            features_df = _calculate_low_value_coin_features(features_df, low_value_data, cross_coin_data)
            
            log.info(f"Added specialized low-value coin sentiment features for {symbol}")
        
        log.debug(f"Handling missing values")
        features_df = features_df.fillna(method='ffill')
        features_df = features_df.fillna(method='bfill')
        
        target_col = f'target_up_{PREDICTION_HORIZON}p'
        features_df = features_df.dropna(subset=[target_col])
        
        if features_df.empty:
            log.warning(f"After dropping NaNs, feature DataFrame is empty for {symbol}")
            return None
            
        if features_df.drop(columns=[target_col], errors='ignore').isnull().values.any():
            log.warning(f"NaN values still detected in features for {symbol} after fill. Replacing with 0.")
            nan_cols = features_df.columns[features_df.isnull().any()].tolist()
            log.warning(f"Columns with NaNs: {nan_cols}")
            features_df = features_df.fillna(0)
            
        log.info(f"Successfully generated {len(features_df)} rows of features for {symbol}")
        return features_df
        
    except Exception as e:
        log.error(f"Error generating features for {symbol}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser(description="Feature Generator for a specific symbol.")
    parser.add_argument('--symbol', type=str, required=True, help="The cryptocurrency symbol (e.g., 'DOGEUSDT').")
    args = parser.parse_args()

    symbol = args.symbol
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
