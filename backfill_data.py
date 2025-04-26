#!/usr/bin/env python3
"""
Backfill Binance OHLCV data safely into the database (with technical indicators and retry logic).
"""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
from tenacity import retry, stop_after_attempt, wait_exponential

from data_collection.binance_client import get_binance_klines, _send_request
from database.db_utils import get_db_pool

log = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.us"

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def safe_get_binance_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    return get_binance_klines(symbol, interval, start_time, end_time, limit)

async def fetch_all_symbols():
    """Fetch all USDT pairs from Binance."""
    data = _send_request("GET", "/api/v3/exchangeInfo")
    symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
    return symbols


def get_latest_prices():
    """Fetch latest prices for all symbols."""
    data = _send_request("GET", "/api/v3/ticker/price")
    return {item['symbol']: float(item['price']) for item in data}


async def backfill_binance_data(start_dt_utc, end_dt_utc, interval="5m"):
    pool = await get_db_pool()

    symbols = await fetch_all_symbols()
    prices = get_latest_prices()

    # Filter sub-$1 symbols only
    sub_usd_symbols = [s for s in symbols if prices.get(s, 9999) < 1]
    log.info(f"Found {len(sub_usd_symbols)} sub-$1 USDT pairs to backfill.")

    async with pool.acquire() as conn:
        for symbol in sub_usd_symbols:
            try:
                log.info(f"Fetching OHLCV for {symbol} from {start_dt_utc} to {end_dt_utc}...")

                current_start = start_dt_utc
                while current_start < end_dt_utc:
                    current_end = min(current_start + timedelta(minutes=5 * 1000), end_dt_utc)

                    klines = safe_get_binance_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=int(current_start.timestamp() * 1000),
                        end_time=int(current_end.timestamp() * 1000)
                    )

                    if not klines:
                        log.warning(f"No data returned for {symbol} between {current_start} and {current_end}.")
                        current_start = current_end
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                    # Calculate indicators
                    df['sma_fast'] = ta.sma(df['close'], length=7)
                    df['sma_slow'] = ta.sma(df['close'], length=25)
                    df['ema_fast'] = ta.ema(df['close'], length=12)
                    df['ema_slow'] = ta.ema(df['close'], length=26)
                    df['rsi_value'] = ta.rsi(df['close'], length=14)
                    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                    df['macd_line'] = macd['MACD_12_26_9']
                    df['macd_signal'] = macd['MACDs_12_26_9']
                    df['macd_hist'] = macd['MACDh_12_26_9']

                    # Insert into DB
                    await conn.executemany(
                        f"""
                        INSERT INTO price_data (
                            symbol, interval, open_time, open, high, low, close, volume, close_time,
                            sma_fast, sma_slow, ema_fast, ema_slow, rsi_value, macd_line, macd_signal, macd_hist,
                            fetched_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9,
                            $10, $11, $12, $13, $14, $15, $16, $17,
                            NOW()
                        )
                        ON CONFLICT (symbol, interval, open_time)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            close_time = EXCLUDED.close_time,
                            sma_fast = EXCLUDED.sma_fast,
                            sma_slow = EXCLUDED.sma_slow,
                            ema_fast = EXCLUDED.ema_fast,
                            ema_slow = EXCLUDED.ema_slow,
                            rsi_value = EXCLUDED.rsi_value,
                            macd_line = EXCLUDED.macd_line,
                            macd_signal = EXCLUDED.macd_signal,
                            macd_hist = EXCLUDED.macd_hist,
                            fetched_at = NOW()
                        ;
                        """,
                        [
                            (
                                symbol, interval,
                                row['open_time'],
                                row['open'], row['high'], row['low'], row['close'], row['volume'], row['close_time'],
                                row['sma_fast'], row['sma_slow'], row['ema_fast'], row['ema_slow'],
                                row['rsi_value'], row['macd_line'], row['macd_signal'], row['macd_hist']
                            )
                            for idx, row in df.iterrows()
                        ]
                    )

                    log.info(f"Inserted {len(df)} records for {symbol}.")

                    # Respect API rate limits
                    await asyncio.sleep(0.5)

                    current_start = current_end

            except Exception as e:
                log.error(f"Error backfilling {symbol}: {e}")

    await pool.close()


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO)

    async def run():
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=60)
        await backfill_binance_data(start_dt, end_dt)

    asyncio.run(run())
