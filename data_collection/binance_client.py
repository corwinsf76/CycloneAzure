# /data_collection/binance_client.py

import logging
import datetime
import pytz
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from database.db_utils import async_df_to_db

log = logging.getLogger(__name__)

# Binance API configuration - adjusted to use the TLD from config
BINANCE_TLD = getattr(config, 'BINANCE_TLD', 'com')
BINANCE_API_BASE = f"https://api.binance.{BINANCE_TLD}/api/v3"

async def get_target_symbols() -> List[str]:
    """Get list of USDT trading pairs from Binance."""
    try:
        headers = {}
        # Add API key to headers if available
        if hasattr(config, 'BINANCE_API_KEY') and config.BINANCE_API_KEY:
            headers['X-MBX-APIKEY'] = config.BINANCE_API_KEY
            
        log.info(f"Using Binance API endpoint: {BINANCE_API_BASE}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BINANCE_API_BASE}/exchangeInfo", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    symbols = [
                        symbol['symbol'] 
                        for symbol in data['symbols']
                        if symbol['quoteAsset'] == 'USDT' and symbol['status'] == 'TRADING'
                    ]
                    return sorted(symbols)
                else:
                    log.error(f"Error fetching exchange info: {response.status}, {await response.text()}")
                    return []
    except Exception as e:
        log.error(f"Error getting target symbols: {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1000,
    start_str: Optional[str] = None,
    end_str: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch kline/candlestick data from Binance API.
    Returns DataFrame with OHLCV data.
    """
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_str:
            params['startTime'] = start_str
        if end_str:
            params['endTime'] = end_str

        headers = {}
        # Add API key to headers if available
        if hasattr(config, 'BINANCE_API_KEY') and config.BINANCE_API_KEY:
            headers['X-MBX-APIKEY'] = config.BINANCE_API_KEY

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BINANCE_API_BASE}/klines", params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    df = pd.DataFrame(data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    if df.empty:
                        return pd.DataFrame()
                    
                    # Convert timestamps to datetime
                    for col in ['open_time', 'close_time']:
                        df[col] = pd.to_datetime(df[col], unit='ms', utc=True)
                    
                    # Convert string values to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    # Select only needed columns
                    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
                    
                    return df
                else:
                    log.error(f"Error fetching klines for {symbol}: {response.status}, {await response.text()}")
                    return pd.DataFrame()
    except Exception as e:
        log.error(f"Error fetching klines for {symbol}: {e}")
        return pd.DataFrame()

async def store_klines(symbol: str, interval: str, klines_df: pd.DataFrame) -> bool:
    """Store klines data in the database."""
    if klines_df.empty:
        return False

    try:
        # Add symbol and interval columns
        klines_df['symbol'] = symbol
        klines_df['interval'] = interval

        # Store in database using async function
        success = await async_df_to_db(klines_df, 'price_data')
        return success
    except Exception as e:
        log.error(f"Error storing klines for {symbol}: {e}")
        return False

async def fetch_and_store_recent_klines(symbol: str, interval: str = "5m", limit: int = 100) -> bool:
    """Fetch and store recent klines for a symbol."""
    klines_df = await fetch_klines(symbol=symbol, interval=interval, limit=limit)
    if not klines_df.empty:
        return await store_klines(symbol, interval, klines_df)
    return False

async def get_market_data(
    symbol: str,
    interval: str = "1h",
    limit: int = 168,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch market data for the dashboard.
    Returns a list of dictionaries with OHLCV data.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        limit: Number of candles to fetch
        start_time: Start time for data fetch
        end_time: End time for data fetch
        
    Returns:
        List of dictionaries with OHLCV data
    """
    try:
        # Convert datetime objects to millisecond timestamps if provided
        start_str = int(start_time.timestamp() * 1000) if start_time else None
        end_str = int(end_time.timestamp() * 1000) if end_time else None
        
        # Fetch klines using existing function
        df = await fetch_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_str=start_str,
            end_str=end_str
        )
        
        if df.empty:
            log.warning(f"No market data available for {symbol} with interval {interval}")
            return []
        
        # Convert DataFrame to list of dictionaries for dashboard consumption
        result = df.to_dict('records')
        
        log.info(f"Fetched {len(result)} {interval} candles for {symbol}")
        return result
        
    except Exception as e:
        log.error(f"Error fetching market data for {symbol}: {e}")
        return []

