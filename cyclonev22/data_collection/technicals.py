# /data_collection/technicals.py

import logging
import pandas as pd
import pandas_ta as ta # Import pandas_ta

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
# from .. import config # Use relative import if running as part of a package

log = logging.getLogger(__name__)

# --- Technical Indicator Calculation ---

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators using pandas_ta and appends them to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data. Must contain columns named
                           'open', 'high', 'low', 'close', 'volume' (case-insensitive).

    Returns:
        pd.DataFrame: Original DataFrame with appended indicator columns, or the
                      original DataFrame if calculation fails or df is empty.
                      Indicator column names are based on pandas_ta defaults but lowercased.
    """
    if df.empty:
        log.warning("Input DataFrame is empty, skipping indicator calculation.")
        return df

    # Ensure required columns exist (case-insensitive check)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df_cols_lower = [col.lower() for col in df.columns]
    if not all(col in df_cols_lower for col in required_cols):
        log.error(f"Input DataFrame missing required columns ({required_cols}). Found: {df.columns}")
        return df

    # Standardize column names to lowercase for pandas_ta compatibility
    df.columns = df.columns.str.lower()

    log.debug(f"Calculating indicators for DataFrame with {len(df)} rows...")

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
        # This appends columns directly to the df
        df.ta.strategy(custom_strategy)

        # --- Rename columns for consistency with planned schema ---
        # pandas_ta names columns like SMA_10, EMA_12, MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9, RSI_14
        rename_map = {
            f"sma_{config.SMA_FAST_PERIOD}": "sma_fast",
            f"sma_{config.SMA_SLOW_PERIOD}": "sma_slow",
            f"ema_{config.EMA_FAST_PERIOD}": "ema_fast",
            f"ema_{config.EMA_SLOW_PERIOD}": "ema_slow",
            f"rsi_{config.RSI_PERIOD}": "rsi_value",
            f"macd_{config.MACD_FAST_PERIOD}_{config.MACD_SLOW_PERIOD}_{config.MACD_SIGNAL_PERIOD}": "macd_line",
            f"macdh_{config.MACD_FAST_PERIOD}_{config.MACD_SLOW_PERIOD}_{config.MACD_SIGNAL_PERIOD}": "macd_hist",
            f"macds_{config.MACD_FAST_PERIOD}_{config.MACD_SLOW_PERIOD}_{config.MACD_SIGNAL_PERIOD}": "macd_signal",
        }
        # Only rename columns that actually exist after calculation
        existing_cols = df.columns
        effective_rename_map = {old: new for old, new in rename_map.items() if old in existing_cols}
        df.rename(columns=effective_rename_map, inplace=True)

        log.debug(f"Successfully calculated indicators. DataFrame shape: {df.shape}")

    except Exception as e:
        log.error(f"Error calculating technical indicators: {e}", exc_info=True)
        # Return original df or handle error as needed

    # Optional: Handle NaNs created by indicator calculations (usually at the start)
    # For ML, might want to fillna or dropna depending on strategy
    # initial_len = len(df)
    # df.dropna(inplace=True) # Or use df.fillna(method='bfill').fillna(method='ffill')
    # if len(df) < initial_len:
    #     log.debug(f"Dropped {initial_len - len(df)} rows with NaNs after indicator calculation.")

    return df


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
