# /data_collection/technicals.py

import logging
import pandas as pd
import pandas_ta as ta  # type: ignore
from pandas import DataFrame, Series
from typing import Optional, cast, Union

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config

log = logging.getLogger(__name__)

# --- Technical Indicator Calculation ---

def calculate_indicators(df: DataFrame) -> DataFrame:
    """Calculate technical indicators for price data."""
    # Standardize column names to lowercase for pandas_ta compatibility
    df = df.copy()  # Work on a copy to avoid modifying input
    df.columns = df.columns.str.lower()

    try:
        # Create a custom strategy for pandas_ta
        custom_strategy = ta.Strategy(
            name="CycloneV2 Indicators",
            description="SMA, EMA, RSI, MACD based on config",
            ta=[
                {"kind": "sma", "length": config.SMA_FAST_PERIOD},
                {"kind": "sma", "length": config.SMA_SLOW_PERIOD},
                {"kind": "ema", "length": config.EMA_FAST_PERIOD},
                {"kind": "ema", "length": config.EMA_SLOW_PERIOD},
                {"kind": "rsi", "length": config.RSI_PERIOD},
                {"kind": "macd", "fast": config.MACD_FAST_PERIOD, "slow": config.MACD_SLOW_PERIOD, "signal": config.MACD_SIGNAL_PERIOD},
            ]
        )

        # Apply the strategy to the DataFrame
        df.ta.strategy(custom_strategy)
        return df
    except Exception as e:
        log.error(f"Error calculating indicators: {e}", exc_info=True)
        return DataFrame()  # Return empty DataFrame instead of None

def calculate_rsi(df: DataFrame, period: int = 14) -> Series:
    """Calculate RSI for a given DataFrame."""
    try:
        rsi_result = ta.rsi(close=df['close'], length=period)
        if isinstance(rsi_result, Series):
            return rsi_result
        elif isinstance(rsi_result, DataFrame):
            return cast(Series, rsi_result.iloc[:, 0])
        else:
            return Series(dtype=float)
    except Exception as e:
        log.error(f"Error calculating RSI: {e}", exc_info=True)
        return Series(dtype=float)

def calculate_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> DataFrame:
    """Calculate MACD for a given DataFrame."""
    try:
        macd_result = ta.macd(close=df['close'], fast=fast, slow=slow, signal=signal)
        if isinstance(macd_result, DataFrame):
            return macd_result
        else:
            return DataFrame()
    except Exception as e:
        log.error(f"Error calculating MACD: {e}", exc_info=True)
        return DataFrame()


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Testing Technical Indicator Calculation ---")

    # Create a sample DataFrame (replace with actual data fetching if needed)
    data = {
        'Open': [100, 101, 102, 101, 103, 104, 105, 106, 105, 107] * 3,
        'High': [102, 103, 103, 102, 104, 105, 106, 107, 106, 108] * 3,
        'Low': [99, 100, 101, 100, 102, 103, 104, 105, 104, 106] * 3,
        'Close': [101, 102, 101, 101, 104, 105, 106, 105, 106, 107] * 3,
        'Volume': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1550, 1800] * 3
    }
    # Need enough data points for indicators to calculate, hence * 3
    sample_df = pd.DataFrame(data)
    # Ensure columns are uppercase initially to test case insensitivity
    sample_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    print(f"\nSample DataFrame (first 5 rows):\n{sample_df.head()}")

    # Calculate indicators
    df_with_indicators = calculate_indicators(sample_df.copy()) # Pass a copy

    if not df_with_indicators.empty and any(col in df_with_indicators.columns for col in ['sma_fast', 'rsi_value', 'macd_line']):
        print(f"\nDataFrame with indicators (first 5 rows):\n{df_with_indicators.head()}")
        print("\nColumns added:")
        added_cols = [col for col in df_with_indicators.columns if col not in sample_df.columns]
        print(added_cols)
        print("\nDataFrame Info:")
        df_with_indicators.info()
    else:
        print("\nFailed to calculate indicators or no indicator columns found.")

    print("\n--- Test Complete ---")
