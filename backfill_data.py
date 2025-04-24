# /backfill_data.py

import logging
import time
import datetime
import pytz
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Any
import json
import asyncio

# Setup project path for imports if running script directly
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
import config
from database.db_utils import async_df_to_db, async_bulk_insert
from data_collection import binance_client, technicals
from binance.client import Client

# --- Configuration ---
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s'
)
log = logging.getLogger(__name__)

# Default interval for data collection
INTERVAL = Client.KLINE_INTERVAL_5MINUTE

async def backfill_binance_data(symbols: List[str], start_dt_utc: datetime.datetime, end_dt_utc: datetime.datetime):
    """Fetches historical klines, calculates TA, and stores in DB."""
    log.info(f"Starting Binance backfill from {start_dt_utc} to {end_dt_utc}")
    
    # Convert timestamps to milliseconds for Binance API
    start_ms = int(start_dt_utc.timestamp() * 1000)
    end_ms = int(end_dt_utc.timestamp() * 1000)
    
    # Calculate time spans and batch sizes
    interval_ms = 5 * 60 * 1000  # 5 minutes in milliseconds
    fetch_limit = 1000  # Binance API limit
    
    tasks = []
    for symbol in symbols:
        tasks.append(_process_symbol(symbol, start_ms, end_ms, interval_ms, fetch_limit))
    
    # Run symbol processing concurrently
    await asyncio.gather(*tasks)
    log.info("Binance backfill complete")

async def _process_symbol(symbol: str, start_ms: int, end_ms: int, interval_ms: int, fetch_limit: int):
    """Process historical data for a single symbol."""
    log.info(f"Processing symbol: {symbol}")
    all_symbol_klines_list = []
    current_start_ms = start_ms
    
    while current_start_ms < end_ms:
        start_dt_log = pd.to_datetime(current_start_ms, unit='ms', utc=True)
        log.debug(f"Fetching klines for {symbol} starting from {start_dt_log}")
        
        try:
            klines_df = await binance_client.fetch_klines(
                symbol=symbol,
                interval=INTERVAL,
                limit=fetch_limit,
                start_str=str(current_start_ms),
                end_str=str(end_ms)
            )

            if klines_df.empty:
                log.debug(f"No more klines found for {symbol}")
                break

            all_symbol_klines_list.append(klines_df)
            log.debug(f"Fetched {len(klines_df)} klines for {symbol}")
            
            # Update start time for next batch
            last_open_time_ms = int(klines_df['open_time'].iloc[-1].timestamp() * 1000)
            current_start_ms = last_open_time_ms + interval_ms
            
            # Add small delay to respect rate limits
            await asyncio.sleep(0.3)
            
        except Exception as e:
            log.error(f"Error fetching klines for {symbol}: {e}", exc_info=True)
            await asyncio.sleep(2)
            break

    if all_symbol_klines_list:
        await _process_klines_data(symbol, all_symbol_klines_list)

async def _process_klines_data(symbol: str, klines_data_list: List[pd.DataFrame]) -> Optional[int]:
    """
    Process klines data for a symbol and insert into database.
    Returns count of processed klines or None on error.
    """
    if not klines_data_list:
        return 0

    try:
        # Combine all dataframes
        combined_df = pd.concat(klines_data_list)
        if combined_df.empty:
            log.warning(f"No klines data to process for {symbol}")
            return 0
            
        # Convert DataFrame to list of dictionaries for database insertion
        parsed_data = []
        for _, row in combined_df.iterrows():
            parsed_kline = {
                'symbol': symbol,
                'interval': config.CANDLE_INTERVAL,
                'open_time': row['open_time'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'close_time': row['close_time'],
                # Technical indicators start as NULL, will be calculated later
                'sma_fast': None,
                'sma_slow': None,
                'ema_fast': None,
                'ema_slow': None,
                'rsi_value': None,
                'macd_line': None,
                'macd_signal': None,
                'macd_hist': None,
                'fetched_at': datetime.datetime.now(datetime.timezone.utc)
            }
            parsed_data.append(parsed_kline)

        # Insert data into database
        result = await async_bulk_insert(
            parsed_data,
            'price_data',
            conflict_fields=['symbol', 'interval', 'open_time'],
            update_fields=['open', 'high', 'low', 'close', 'volume', 'close_time']
        )
        
        if result is False:
            log.error(f"Failed to insert {len(parsed_data)} klines for {symbol}")
            return None
            
        return len(parsed_data)
    except Exception as e:
        log.error(f"Error processing klines data for {symbol}: {e}")
        return None

async def main():
    """Main function to run backfill process."""
    # Get target symbols
    symbols = await binance_client.get_target_symbols()
    if not symbols:
        log.error("No symbols found to process")
        return
    
    # Set date range
    end_dt = datetime.datetime.now(pytz.UTC)
    start_dt = end_dt - datetime.timedelta(days=config.BACKFILL_DAYS)
    
    await backfill_binance_data(symbols, start_dt, end_dt)

if __name__ == "__main__":
    asyncio.run(main())