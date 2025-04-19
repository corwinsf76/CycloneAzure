# /backfill_data.py

import logging
import time
import datetime
import pytz
import pandas as pd
from typing import List, Optional, Tuple # Added Tuple
import json # Added json
import numpy as np # Added numpy

# Setup project path for imports if running script directly
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming backfill_data.py is in the project root (cyclonev2/)
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # Use print before logging configured if needed, or configure basic logging early
    # print(f"Added project root to sys.path: {project_root}")

# Import project modules
import config
from database import db_utils
from data_collection import binance_client, technicals # Only import needed modules
# Import Client for constants like KLINE_INTERVAL_5MINUTE
from binance.client import Client

# --- Configuration ---
logging.basicConfig(level=config.LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')
log = logging.getLogger(__name__)

# Define backfill period and interval from config or defaults
BACKFILL_DAYS = config.get_env_variable("BACKFILL_DAYS_PRICE", 90, var_type=int) # Optional: Specific config var
# Use the same interval as the trading logic/feature generation
# Assuming 5 minutes based on previous analysis of other files
INTERVAL = Client.KLINE_INTERVAL_5MINUTE
# Convert interval string to Timedelta for loop update calculation
try:
    INTERVAL_TIMEDELTA = pd.to_timedelta(INTERVAL)
except ValueError:
    log.error(f"Invalid interval string '{INTERVAL}' for pd.to_timedelta. Using 5 minutes as fallback.")
    INTERVAL = Client.KLINE_INTERVAL_5MINUTE # Fallback to known good value
    INTERVAL_TIMEDELTA = pd.to_timedelta('5m')


# --- Helper Functions ---

def format_binance_time(dt_obj: datetime.datetime) -> int:
    """Converts a datetime object to Binance compatible millisecond timestamp."""
    return int(dt_obj.timestamp() * 1000)

# --- Backfill Functions ---

def backfill_binance_data(symbols: List[str], start_dt_utc: datetime.datetime, end_dt_utc: datetime.datetime):
    """Fetches historical klines, calculates TA, and stores in DB."""
    log.info(f"Starting Binance backfill for {len(symbols)} symbols from {start_dt_utc} to {end_dt_utc}...")
    if not db_utils.engine:
        log.critical("Database engine not configured. Cannot backfill Binance data.")
        return

    interval_ms = int(INTERVAL_TIMEDELTA.total_seconds() * 1000)
    total_inserted_count = 0

    for i, symbol in enumerate(symbols):
        log.info(f"Processing symbol {i+1}/{len(symbols)}: {symbol}")
        all_symbol_klines_list = [] # Use a list to collect DataFrame chunks
        current_start_ms = format_binance_time(start_dt_utc)
        end_ms = format_binance_time(end_dt_utc)
        fetch_limit = 1000 # Binance API limit per request

        while current_start_ms < end_ms:
            start_dt_log = pd.to_datetime(current_start_ms, unit='ms', utc=True)
            log.debug(f"Fetching klines for {symbol} starting from {start_dt_log}")
            try:
                klines_df = binance_client.fetch_klines(
                    symbol=symbol,
                    interval=INTERVAL,
                    limit=fetch_limit,
                    start_str=str(current_start_ms), # Use string representation of ms timestamp
                    end_str=str(end_ms) # Fetch up to the end time in each request
                )

                if klines_df.empty:
                    log.debug(f"No more klines found for {symbol} starting at {start_dt_log}.")
                    break # No more data for this symbol in the range

                # Ensure data doesn't exceed the absolute end time (optional, Binance end_str might handle this)
                klines_df = klines_df[klines_df['open_time'] <= end_dt_utc]
                if klines_df.empty:
                     log.debug(f"Klines fetched but all were after end_dt_utc {end_dt_utc}.")
                     break

                # --- Corrected Append ---
                all_symbol_klines_list.append(klines_df)
                log.debug(f"Fetched {len(klines_df)} klines for {symbol} ending at {klines_df['open_time'].iloc[-1]}")

                # --- Corrected Loop Update ---
                # Get the timestamp of the last kline fetched
                last_open_time_ms = int(klines_df['open_time'].iloc[-1].timestamp() * 1000)
                # Set the start for the next fetch to be one interval after the last fetched candle's open time
                current_start_ms = last_open_time_ms + interval_ms

                # Add a small delay to respect API limits
                time.sleep(0.3) # Adjusted delay slightly

            except Exception as fetch_err:
                log.error(f"Error fetching klines for {symbol} starting at {start_dt_log}: {fetch_err}", exc_info=True)
                # Decide how to handle errors: break, continue, retry?
                time.sleep(2) # Pause longer after an error
                break # Example: Stop fetching for this symbol on error

        # --- Corrected Data Processing Location ---
        if all_symbol_klines_list:
            # Combine all chunks for the current symbol
            combined_symbol_df = pd.concat(all_symbol_klines_list, ignore_index=True)
            # Remove duplicates just in case fetch windows overlapped slightly
            combined_symbol_df.drop_duplicates(subset=['open_time'], keep='first', inplace=True)
            # Filter one last time to ensure no data beyond end_dt_utc
            combined_symbol_df = combined_symbol_df[combined_symbol_df['open_time'] <= end_dt_utc]
            # Sort to ensure chronological order before TA calculation
            combined_symbol_df.sort_values(by='open_time', inplace=True)

            if combined_symbol_df.empty:
                 log.info(f"No valid klines remained after combining/filtering for {symbol}.")
                 continue # Skip to next symbol

            log.info(f"Combined {len(combined_symbol_df)} unique klines for {symbol}. Calculating TA...")

            # Calculate indicators
            klines_with_ta = technicals.calculate_indicators(combined_symbol_df) # Pass the combined DataFrame

            if not klines_with_ta.empty:
                # Prepare for DB insertion (match schema columns)
                klines_with_ta['symbol'] = symbol
                klines_with_ta['interval'] = INTERVAL # Store the interval used

                # Select only columns present in the price_data table definition
                db_cols = [c.name for c in db_utils.price_data.columns if c.name != 'id' and c.name != 'fetched_at']

                # Filter DataFrame columns to match DB schema exactly
                # Ensure all base columns needed by DB are present before selecting
                required_db_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'symbol', 'interval']
                missing_req_cols = [col for col in required_db_cols if col not in klines_with_ta.columns]
                if missing_req_cols:
                     log.error(f"DataFrame for {symbol} is missing required DB columns: {missing_req_cols}. Skipping DB store.")
                     continue

                # Select existing columns that are part of the schema
                cols_to_select = [col for col in db_cols if col in klines_with_ta.columns]
                klines_to_store = klines_with_ta[cols_to_select]

                try:
                    # Convert potential Pandas NaNs (from TA) to None for DB compatibility
                    klines_to_store_db = klines_to_store.replace({np.nan: None})

                    # Use df_to_db for bulk insert
                    # 'append' relies on DB unique constraints (symbol, interval, open_time) to prevent duplicates
                    # For a clean backfill, consider deleting old data in the range first, or using a proper upsert if DB supports it.
                    db_utils.df_to_db(klines_to_store_db, db_utils.price_data.name, if_exists='append', index=False)

                    # This count is approximate as duplicates might be rejected by DB constraint
                    inserted_rows_estimate = len(klines_to_store_db)
                    total_inserted_count += inserted_rows_estimate
                    log.info(f"Attempted to store {inserted_rows_estimate} klines with TA for {symbol}.")

                except Exception as db_err:
                    log.error(f"Error writing price data to DB for {symbol}: {db_err}", exc_info=True)
            else:
                log.warning(f"No data remained for {symbol} after calculating indicators (maybe requires more history?).")
        else:
            log.info(f"No klines fetched or combined for {symbol}.")

    # --- End of symbol loop ---
    log.info(f"Binance backfill finished. Attempted to insert approximately {total_inserted_count} total rows across all symbols.")


# --- Main Execution ---
if __name__ == "__main__":
    log.info(f"--- Starting Binance Price Data Backfill Script ({BACKFILL_DAYS} days) ---")

    # --- Determine Date Range ---
    end_dt = datetime.datetime.now(pytz.utc)
    start_dt = end_dt - pd.Timedelta(days=BACKFILL_DAYS)
    log.info(f"Backfill Range: {start_dt} -> {end_dt}")

    # --- Initialize DB ---
    if not db_utils.init_db():
        log.critical("Database initialization failed. Exiting backfill.")
        sys.exit(1)
    log.info("Database schema initialized successfully.")

    # --- Get Symbols ---
    try:
        target_symbols = binance_client.get_target_symbols()
        # Optionally filter symbols further if needed
        # target_symbols = [s for s in target_symbols if s.endswith('USDT')]
    except Exception as e:
        log.critical(f"Failed to get target symbols from Binance: {e}. Exiting.")
        sys.exit(1)

    if not target_symbols:
        log.warning("No target symbols found. Nothing to backfill.")
        sys.exit(0)

    # --- Run Backfill ---
    try:
        backfill_binance_data(target_symbols, start_dt, end_dt)
    except Exception as e:
        log.critical(f"An unexpected error occurred during the backfill process: {e}", exc_info=True)

    log.info("--- Binance Price Data Backfill Script Finished ---")
    print("\nBinance data backfill process complete. Check logs for details and errors.")