# /feature_engineering/feature_generator.py

import logging
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex
from typing import List, Dict, Optional, Any, Union, cast, TypeVar
import numpy as np
from sqlalchemy.sql import select, and_, or_, text
import datetime
import pytz

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils # Import the db_utils module to access table objects and engine

log = logging.getLogger(__name__)

# Type variables for pandas operations
T = TypeVar('T', bound=Union[Series, DataFrame])

# --- Constants ---
# Define feature calculation parameters (can also be moved to config if needed)
PRICE_LAG_PERIODS = config.FEATURE_LAG_PERIODS # e.g., 20
ROLLING_WINDOWS = ['1h', '4h', '12h', '24h'] # Pandas offset strings for rolling stats
SENTIMENT_WINDOWS = [config.SENTIMENT_AGG_WINDOW_SHORT, config.SENTIMENT_AGG_WINDOW_LONG] # e.g., ['1h', '24h']
PREDICTION_HORIZON = config.PREDICTION_HORIZON_PERIODS # e.g., 3 periods (15 mins for 5m interval)

# --- Helper Functions ---

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

def _calculate_rolling_sentiment(sentiment_df: DataFrame, price_timestamps: DatetimeIndex, window: str) -> DataFrame:
    """Calculates rolling average sentiment scores aligned with price timestamps."""
    sentiment_df = _ensure_datetime_index(sentiment_df)
    sentiment_df['sentiment_score'] = pd.to_numeric(sentiment_df['sentiment_score'], errors='coerce')
    
    # Calculate rolling stats
    resampled = sentiment_df.resample(rule='1Min')  # Remove 'on' parameter as index is already set
    sentiment_mean = cast(Series, resampled['sentiment_score'].mean())
    sentiment_count = cast(Series, resampled['sentiment_score'].count())
    
    rolling_mean = cast(Series, sentiment_mean.rolling(window=window, min_periods=1).mean())
    rolling_count = cast(Series, sentiment_count.rolling(window=window, min_periods=1).sum())
    
    # Create result DataFrame with both metrics
    result = pd.DataFrame({
        f'sent_avg_{window}': rolling_mean,
        f'sent_count_{window}': rolling_count
    })
    
    return cast(DataFrame, result.reindex(price_timestamps))

def _calculate_target(price_df: DataFrame, periods: int) -> DataFrame:
    """
    Calculate binary target variable based on future price movement.
    
    Args:
        price_df: DataFrame with price data
        periods: Number of periods to look ahead
        
    Returns:
        DataFrame with binary target column
    """
    future_returns = price_df['close'].pct_change(periods=periods).shift(-periods)
    target_col_name = f'target_up_{periods}p'
    
    target = pd.DataFrame({
        target_col_name: (future_returns > 0).astype(np.int32)
    }, index=price_df.index)
    
    return target

# --- Main Feature Generation Function ---

def generate_features_for_symbol(
    symbol: str,
    end_time_utc: Optional[datetime.datetime],
    history_duration: Optional[datetime.timedelta] = datetime.timedelta(days=30)
) -> Optional[DataFrame]:
    """Generate features for a symbol up to specified end time."""
    if end_time_utc is None:
        log.error("end_time_utc must be provided")
        return None
        
    if history_duration is None:
        history_duration = datetime.timedelta(days=30)
        
    if not end_time_utc.tzinfo or end_time_utc.tzinfo.utcoffset(end_time_utc) != datetime.timedelta(0):
        log.error("end_time_utc must be timezone-aware UTC.")
        return None
        
    if db_utils.engine is None:
        log.error("Feature Generation: DB engine not configured.")
        return None

    start_time_utc = end_time_utc - history_duration
    interval = '5m' # Assuming 5-minute interval based on config fetch interval

    log.info(f"Generating features for {symbol} from {start_time_utc} to {end_time_utc}")

    # --- 1. Fetch Raw Data ---
    price_df = None
    sentiment_df = None
    try:
        # Use the engine directly for pd.read_sql when passing SQLAlchemy selectables
        engine = db_utils.engine # Get the engine instance

        # Fetch Price Data
        price_query = select(db_utils.price_data).where(
            db_utils.price_data.c.symbol == symbol,
            db_utils.price_data.c.interval == interval,
            db_utils.price_data.c.open_time >= start_time_utc,
            db_utils.price_data.c.open_time <= end_time_utc # Include end time candle
        ).order_by(db_utils.price_data.c.open_time)
        # *** FIX: Pass engine, not session, to pd.read_sql ***
        price_df = pd.read_sql(price_query, engine, index_col='open_time', parse_dates=['open_time'])
        if price_df.empty:
            log.warning(f"No price data found for {symbol} in the specified range.")
            # Still return None here, as features cannot be generated without price data
            return None
        # Ensure index is timezone-aware UTC
        price_df = _ensure_datetime_index(price_df)
        log.info(f"Fetched {len(price_df)} price data points for {symbol}.")

        # Fetch Sentiment Data (combine all sources for simplicity here)
        sentiment_query = select(
            db_utils.sentiment_analysis_results.c.analyzed_at,
            db_utils.sentiment_analysis_results.c.sentiment_score
            # TODO: Filter sentiment by symbol (requires joins/schema changes)
        ).where(
            db_utils.sentiment_analysis_results.c.analyzed_at >= start_time_utc,
            db_utils.sentiment_analysis_results.c.analyzed_at <= end_time_utc
        ).order_by(db_utils.sentiment_analysis_results.c.analyzed_at)
        # *** FIX: Pass engine, not session, to pd.read_sql ***
        sentiment_df = pd.read_sql(sentiment_query, engine, index_col='analyzed_at', parse_dates=['analyzed_at'])
        # Ensure index is timezone-aware UTC
        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df = _ensure_datetime_index(sentiment_df)
        log.info(f"Fetched {len(sentiment_df)} sentiment data points (all symbols).") # Note: All symbols

    except Exception as e:
        log.error(f"Database error fetching data for {symbol}: {e}", exc_info=True)
        return None

    # --- 2. Feature Calculation ---
    # Ensure price_df is not None before proceeding
    if price_df is None or price_df.empty:
         log.error(f"Cannot calculate features for {symbol} due to missing price data.")
         return None

    features = pd.DataFrame(index=price_df.index)
    features['symbol'] = symbol # Add symbol identifier

    # a) Price/Volume Lags & Returns
    log.debug("Calculating price/volume features...")
    for lag in range(1, PRICE_LAG_PERIODS + 1):
        features[f'lag_close_{lag}'] = price_df['close'].shift(lag)
        features[f'lag_volume_{lag}'] = price_df['volume'].shift(lag)
        features[f'return_{lag}'] = price_df['close'].pct_change(periods=lag)

    # b) Rolling Price/Volume Statistics
    for window in ROLLING_WINDOWS:
        # closed='left' ensures data up to (but not including) the current candle is used
        features[f'roll_avg_close_{window}'] = cast(Series, price_df['close'].rolling(window, closed='left').mean())
        features[f'roll_std_close_{window}'] = cast(Series, price_df['close'].rolling(window, closed='left').std())
        features[f'roll_avg_vol_{window}'] = cast(Series, price_df['volume'].rolling(window, closed='left').mean())

    # c) Technical Indicators (already fetched, just use them)
    log.debug("Adding technical indicator features...")
    indicator_cols = ['sma_fast', 'sma_slow', 'ema_fast', 'ema_slow', 'rsi_value', 'macd_line', 'macd_signal', 'macd_hist']
    for col in indicator_cols:
        if col in price_df.columns:
            features[col] = price_df[col]
            # Create diff/cross features only if both components exist
            if col == 'sma_slow' and 'sma_fast' in features.columns: features['sma_diff'] = features['sma_fast'] - features['sma_slow']
            if col == 'ema_slow' and 'ema_fast' in features.columns: features['ema_diff'] = features['ema_fast'] - features['ema_slow']
            if col == 'macd_signal' and 'macd_line' in features.columns: features['macd_diff'] = features['macd_line'] - features['macd_signal']

    # d) Rolling Sentiment Features
    log.debug("Calculating rolling sentiment features...")
    features = _ensure_datetime_index(features)
    idx = cast(DatetimeIndex, features.index)
    if sentiment_df is not None and not sentiment_df.empty: # Check if sentiment_df exists
         # Ensure sentiment_df index is unique if duplicates exist (e.g., keep last)
         sentiment_df = sentiment_df[~sentiment_df.index.duplicated(keep='last')]
         for window in SENTIMENT_WINDOWS:
             rolling_sent = _calculate_rolling_sentiment(sentiment_df.copy(), idx, window) # Pass copy
             features = pd.concat([features, rolling_sent], axis=1)
    else:
         # Add NaN columns if no sentiment data
         log.warning("No sentiment data found for feature calculation.")
         for window in SENTIMENT_WINDOWS:
             features[f'sent_avg_{window}'] = np.nan
             features[f'sent_count_{window}'] = np.nan


    # e) Time-based Features
    log.debug("Calculating time-based features...")
    # Fix DataFrame index type issues
    features['hour'] = idx.hour.astype(np.int32)
    features['dayofweek'] = idx.dayofweek.astype(np.int32)
    features['minute'] = idx.minute.astype(np.int32)
    # Cyclical features
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['day_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)

    # --- 3. Calculate Target Variable ---
    log.debug("Calculating target variable...")
    target = _calculate_target(price_df, periods=PREDICTION_HORIZON)
    features = pd.concat([features, target], axis=1)

    # --- 4. Cleanup ---
    # Drop initial rows with NaNs created by lags/rolling features/target
    # Ensure target column name is correctly referenced
    target_col_name = f'target_up_{PREDICTION_HORIZON}p'
    min_required_periods = PRICE_LAG_PERIODS # Minimum needed for lags
    # Also consider longest rolling window, adjust drop logic if needed
    initial_len = len(features)
    # Drop rows where longest lag OR target is NaN
    features.dropna(subset=[f'lag_close_{PRICE_LAG_PERIODS}', target_col_name], inplace=True)
    final_len = len(features)
    log.info(f"Feature generation complete for {symbol}. Shape: {features.shape}. Dropped {initial_len - final_len} initial/final rows with NaNs.")

    # Optional: Fill remaining NaNs in features if any (e.g., std dev on flat data, initial sentiment NaNs)
    # Check for any remaining NaNs before filling
    if features.isnull().values.any():
         nan_cols = features.columns[features.isnull().any()].tolist()
         log.warning(f"Features still contain NaNs after initial drop: {nan_cols}. Imputing with 0.")
         features.fillna(0, inplace=True) # Or use more sophisticated imputation

    return cast(DataFrame, features)


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')

    print("--- Testing Feature Generator ---")
    # Note: This test requires data to be present in the database.
    # Ensure data collection has run first.

    test_symbol = 'BTCUSDT' # Use a symbol likely to have data
    # Use a recent end time, assuming data exists up to now
    end_time = datetime.datetime.now(pytz.utc)
    # Fetch last 7 days for feature calculation example
    history = pd.Timedelta(days=7)

    print(f"Generating features for {test_symbol} up to {end_time} with {history} history...")

    # Ensure DB is initialized (if running standalone)
    # db_utils.init_db() # Uncomment if needed, ensure config is correct

    features_df = generate_features_for_symbol(test_symbol, end_time, history_duration=history)

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
        # Check for any remaining NaNs
        if features_df.isnull().values.any():
             print("\nWarning: NaNs detected in final feature DataFrame!")
             print(features_df.isnull().sum())
        else:
             print("\nNo NaNs detected in final feature DataFrame.")

    elif features_df is not None and features_df.empty:
         print(f"\nFeature generation ran but resulted in an empty DataFrame (possibly due to NaNs or insufficient history).")
    else:
        print(f"\nFailed to generate features for {test_symbol}. Check logs and database connection/data.")

    print("\n--- Test Complete ---")
