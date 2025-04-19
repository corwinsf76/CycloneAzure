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
    print(f"Added project root to sys.path: {project_root}")

# Import project modules
import config
from database import db_utils
from data_collection import binance_client, technicals, cryptonews_client, reddit_client, twitter_client # Added reddit/twitter
from sentiment_analysis import analyzer
# Import the sentiment job logic helper from scheduler
try:
    # Assuming the logic function is reusable and doesn't depend on scheduler state
    from orchestration.scheduler import run_sentiment_analysis_job_logic
except ImportError as e:
    log = logging.getLogger(__name__) # Need logger if import fails early
    log.error(f"Could not import run_sentiment_analysis_job_logic from scheduler: {e}. Sentiment analysis step will be skipped.")
    # Define a dummy function if import fails to avoid NameError later
    def run_sentiment_analysis_job_logic():
        log = logging.getLogger(__name__) # Need logger inside dummy too
        log.warning("Sentiment analysis logic function not available. Skipping analysis.")
        pass


# --- Configuration ---
logging.basicConfig(level=config.LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')
log = logging.getLogger(__name__)

# Define backfill period
BACKFILL_DAYS = 90 # Approx 3 months (as requested, matching default config)
INTERVAL = Client.KLINE_INTERVAL_5MINUTE # Match the interval used for features/trading

# --- Helper Functions ---

def format_binance_time(dt_obj: datetime.datetime) -> int:
    """Converts a datetime object to Binance compatible millisecond timestamp."""
    return int(dt_obj.timestamp() * 1000)

# Removed format_cryptonews_date as we'll use relative 'lastXdays' or rely on client formatting if needed

# --- Backfill Functions ---

def backfill_binance_data(symbols: List[str], start_dt_utc: datetime.datetime, end_dt_utc: datetime.datetime):
    """Fetches historical klines, calculates TA, and stores in DB."""
    log.info(f"Starting Binance backfill for {len(symbols)} symbols from {start_dt_utc} to {end_dt_utc}...")
    if not db_utils.engine:
        log.critical("Database engine not configured. Cannot backfill Binance data.")
        return

    total_inserted_count = 0
    for i, symbol in enumerate(symbols):
        log.info(f"Processing symbol {i+1}/{len(symbols)}: {symbol}")
        all_symbol_klines = []
        current_start_ms = format_binance_time(start_dt_utc)
        end_ms = format_binance_time(end_dt_utc)
        fetch_limit = 1000 # Binance API limit per request

        while current_start_ms < end_ms:
            log.debug(f"Fetching klines for {symbol} starting from {pd.to_datetime(current_start_ms, unit='ms', utc=True)}")
            try:
                klines_df = binance_client.fetch_klines(
                    symbol=symbol,
                    interval=INTERVAL,
                    limit=fetch_limit,
                    start_str=str(current_start_ms), # Use string representation of ms timestamp
                    end_str=str(end_ms) # Fetch up to the end time in each request
                )

                if klines_df.empty:
                    log.debug(f"No more klines found for {symbol} starting at {current_start_ms}.")
                    break # No more data for this symbol in the range

                # Filter out data beyond the requested end time precisely
                klines_df = klines_df[klines_df['open_time'] <= end_dt_utc]
                if klines_df.empty:
                    break

                all_symbol_klines.append(kli