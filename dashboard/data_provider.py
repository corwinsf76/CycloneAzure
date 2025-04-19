# /dashboard/data_provider.py

import logging
import pandas as pd
import datetime
import pytz
from sqlalchemy.sql import select, desc, text
from typing import Dict, List, Optional, Any, Tuple, cast
from pandas import DataFrame, DatetimeIndex
from sqlalchemy import create_engine
import sys
import os

# Add the project root directory to PYTHONPATH dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils # Import the db_utils module to access engine, table objects and session
from trading import trader # To get disabled symbols list
from database.db_utils import engine

log = logging.getLogger(__name__)

# --- Helper Functions ---

def _ensure_datetime_index(df: DataFrame) -> DataFrame:
    """Ensure DataFrame has a timezone-aware datetime index."""
    if not isinstance(df.index, DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Use type checking to avoid 'UnknownIndex' type errors
    if hasattr(df.index, 'tz_localize') and hasattr(df.index, 'tz_convert'):
        idx = cast(DatetimeIndex, df.index)
        if idx.tz is None:
            df.index = idx.tz_localize(pytz.UTC)
        elif idx.tz != pytz.UTC:
            df.index = idx.tz_convert(pytz.UTC)
    else:
        # Fall back for non-DatetimeIndex
        log.warning("Index doesn't support timezone operations - creating new DatetimeIndex")
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(pytz.UTC)
    
    return df

# --- Data Fetching Functions for Dashboard ---

def get_overview_data() -> Dict[str, Any]:
    """Fetches key metrics for the overview section."""
    data = {
        "pnl_total": 0.0, # TODO: Calculate overall PnL (needs careful tracking)
        "open_positions_count": 0, # TODO: Get from portfolio manager state or DB
        "cash": config.INITIAL_CAPITAL_USD, # TODO: Get current cash from portfolio manager state
        "total_value": config.INITIAL_CAPITAL_USD, # TODO: Get from portfolio manager state
        "peak_value": config.INITIAL_CAPITAL_USD, # TODO: Get from portfolio manager state
        "halt_status": False, # TODO: Get from portfolio manager state
        "latest_trades": pd.DataFrame(),
        "error_alerts": [] # TODO: Implement system for tracking/displaying errors
    }
    if not db_utils.engine:
        log.error("Database engine not available for get_overview_data.")
        return data # Return defaults if DB not configured

    try:
        # Use the engine directly with pd.read_sql when passing SQLAlchemy selectables
        engine = db_utils.engine
        trade_query = select(db_utils.trade_log).order_by(desc(db_utils.trade_log.c.timestamp)).limit(10)
        data['latest_trades'] = pd.read_sql(trade_query, engine, parse_dates=['timestamp']) # Use engine

        # Convert timestamp to user's local timezone for display? Or keep UTC? Let's keep UTC for now.
        if not data['latest_trades'].empty and data['latest_trades']['timestamp'].dt.tz is None:
             # Ensure timezone aware if read directly without timezone info
             data['latest_trades']['timestamp'] = data['latest_trades']['timestamp'].dt.tz_localize(pytz.utc)
        elif not data['latest_trades'].empty:
             data['latest_trades']['timestamp'] = data['latest_trades']['timestamp'].dt.tz_convert(pytz.utc) # Ensure UTC
        # data['latest_trades']['timestamp'] = data['latest_trades']['timestamp'].dt.tz_convert('America/Los_Angeles') # Example conversion

        # TODO: Query portfolio state if persisted in DB, otherwise needs access to live PortfolioManager instance
        # This is a challenge for a separate dashboard process. Might need IPC or a shared state DB.
        # For now, returning defaults or placeholders.

    except Exception as e:
        log.error(f"Error fetching overview data from DB: {e}", exc_info=True)

    # Get disabled symbols from trader module (in-memory state)
    # Ensure trader module is imported and state is accessible
    try:
        data["disabled_symbols"] = list(trader.DISABLED_SYMBOLS)
    except AttributeError: # Handle case where trader or DISABLED_SYMBOLS might not be initialized yet
        log.warning("Could not access trader.DISABLED_SYMBOLS state.")
        data["disabled_symbols"] = []

    # Get halt status from portfolio manager (needs access to the instance)
    # data["halt_status"] = portfolio_manager_instance.halt_trading_flag # Requires instance access

    return data


def get_market_data(symbol: str, interval: str = '5m') -> DataFrame:
    """Get market data for a symbol."""
    market_df = pd.read_sql_query(
        f"SELECT * FROM price_data WHERE symbol = '{symbol}' AND interval = '{interval}'",
        engine,
        index_col='open_time',
        parse_dates=['open_time']
    )
    return _ensure_datetime_index(market_df)


def get_sentiment_data(symbol: str) -> DataFrame:
    """Get sentiment data for a symbol."""
    sentiment_df = pd.read_sql_query(
        f"SELECT * FROM sentiment_analysis_results WHERE symbol = '{symbol}'",
        engine,
        index_col='analyzed_at',
        parse_dates=['analyzed_at']
    )
    return _ensure_datetime_index(sentiment_df)


def get_pnl_data(symbol: str) -> DataFrame:
    """Get PnL data for a symbol."""
    pnl_df = pd.read_sql_query(
        f"SELECT * FROM trade_pnl WHERE symbol = '{symbol}'",
        engine,
        index_col='timestamp',
        parse_dates=['timestamp']
    )
    return _ensure_datetime_index(pnl_df)


def get_trading_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches open positions and trade history."""
    open_positions_df = pd.DataFrame(columns=['symbol', 'quantity', 'entry_price', 'entry_time']) # Placeholder
    trade_history_df = pd.DataFrame()

    # TODO: Get open positions from live PortfolioManager instance or persisted state
    # This is a placeholder implementation fetching from trade log (not accurate for open positions)
    # open_positions = portfolio_manager_instance.get_all_positions()
    # open_positions_df = pd.DataFrame.from_dict(open_positions, orient='index')

    if not db_utils.engine:
        log.error("Database engine not available for get_trading_data.")
        return open_positions_df, trade_history_df

    try:
        engine = db_utils.engine
        # Fetch recent trade history
        history_query = select(db_utils.trade_log).order_by(desc(db_utils.trade_log.c.timestamp)).limit(100) # Fetch last 100 trades
        trade_history_df = pd.read_sql(history_query, engine, parse_dates=['timestamp']) # Use engine

        # Ensure proper timezone handling with type checking
        if not trade_history_df.empty:
            timestamp_col = trade_history_df['timestamp']
            # Check if the timestamp column is a datetime type and has timezone methods
            if hasattr(timestamp_col, 'dt') and hasattr(timestamp_col.dt, 'tz'):
                if timestamp_col.dt.tz is None:
                    # Ensure timezone aware if read directly without timezone info
                    trade_history_df['timestamp'] = timestamp_col.dt.tz_localize(pytz.utc)
                else:
                    # Convert to UTC if it has a different timezone
                    trade_history_df['timestamp'] = timestamp_col.dt.tz_convert(pytz.utc)
            else:
                # Handle non-datetime column by converting it first
                trade_history_df['timestamp'] = pd.to_datetime(timestamp_col).dt.tz_localize(pytz.utc)

        log.debug(f"Fetched {len(trade_history_df)} trade history records.")

    except Exception as e:
        log.error(f"Error fetching trading data from DB: {e}", exc_info=True)

    return open_positions_df, trade_history_df


def get_performance_data() -> pd.DataFrame:
    """Fetches data needed for performance analysis (e.g., PnL over time)."""
    pnl_df = pd.DataFrame()
    if not db_utils.engine:
        log.error("Database engine not available for get_performance_data.")
        return pnl_df

    try:
        engine = db_utils.engine
        pnl_query = select(
                db_utils.trade_log.c.timestamp,
                db_utils.trade_log.c.pnl
            ).where(
                db_utils.trade_log.c.trade_type == 'SELL',
                db_utils.trade_log.c.pnl.isnot(None) # Ensure PnL is calculated
            ).order_by(db_utils.trade_log.c.timestamp)
        pnl_df = pd.read_sql(pnl_query, engine, index_col='timestamp', parse_dates=['timestamp']) # Use engine

        # Adding proper attribute checks to avoid UnknownIndex errors
        if not pnl_df.empty:
            if hasattr(pnl_df.index, 'tz_localize') and isinstance(pnl_df.index, DatetimeIndex) and pnl_df.index.tz is None:
                # Ensure index is timezone-aware UTC if read without timezone info
                pnl_df.index = pnl_df.index.tz_localize(pytz.utc)
            elif hasattr(pnl_df.index, 'tz_convert'):
                pnl_df.index = pnl_df.index.tz_convert(pytz.utc) # Ensure UTC
            
            # Calculate cumulative PnL after ensuring proper index
            pnl_df['cumulative_pnl'] = pnl_df['pnl'].cumsum()
            
        log.debug(f"Fetched {len(pnl_df)} PnL records for performance analysis.")
    except Exception as e:
        log.error(f"Error fetching performance data from DB: {e}", exc_info=True)
    return pnl_df


def get_config_settings() -> Dict[str, Any]:
    """Returns current trading configuration settings."""
    # Reads directly from the (potentially updated at runtime) config module
    # NOTE: This assumes the dashboard process shares the same memory space
    # or can somehow access the potentially modified config values.
    # This might not work reliably if trader/dashboard are separate processes.
    try:
        return {
            "trade_capital_percentage": config.TRADE_CAPITAL_PERCENTAGE,
            "stop_loss_pct": config.STOP_LOSS_PCT,
            "take_profit_pct": config.TAKE_PROFIT_PCT,
            "portfolio_drawdown_pct": config.PORTFOLIO_DRAWDOWN_PCT,
            # Add other relevant config values
        }
    except Exception as e:
        log.error(f"Error reading config settings: {e}")
        return {} # Return empty dict on error

def get_target_symbols_list() -> List[str]:
     """Gets the list of symbols the bot might trade."""
     # This could be dynamic based on get_target_symbols or a fixed list
     # For dropdowns, maybe fetch all symbols with recent price data?
     symbols = []
     if not db_utils.engine:
         log.error("Database engine not available for get_target_symbols_list.")
         return ['TESTSYMBOL1', 'TESTSYMBOL2'] # Fallback

     try:
         engine = db_utils.engine
         # Query distinct symbols from recent price data
         query = select(db_utils.price_data.c.symbol.distinct()).where(
             db_utils.price_data.c.open_time >= datetime.datetime.now(pytz.utc) - pd.Timedelta(days=1)
         ).order_by(db_utils.price_data.c.symbol)
         # Use engine for connection in read_sql
         with engine.connect() as connection:
             symbols_df = pd.read_sql(query, connection) # Use connection from engine
             symbols = symbols_df['symbol'].tolist()
     except Exception as e:
         log.error(f"Error fetching distinct symbols from DB: {e}", exc_info=True)
     return symbols if symbols else ['TESTSYMBOL1', 'TESTSYMBOL2'] # Fallback


def get_market_insights_data(timeframe: str = '24h') -> dict:
    """
    Fetches market insights data from all API sources.
    
    Args:
        timeframe: Time window for data, e.g., '24h', '7d'
    
    Returns:
        Dictionary containing DataFrames for each data source
    """
    try:
        if not engine:
            log.error("Database engine not available for get_market_insights_data")
            return {
                'cryptopanic': pd.DataFrame(),
                'alphavantage': pd.DataFrame(),
                'coingecko': pd.DataFrame()
            }
            
        with engine.connect() as conn:
            # Calculate time window
            end_time = datetime.datetime.now(pytz.utc)
            hours = int(timeframe.replace('h', '')) if 'h' in timeframe else 24 * int(timeframe.replace('d', ''))
            start_time = end_time - datetime.timedelta(hours=hours)
            
            # CryptoPanic sentiment data
            sentiment_query = text("""
                SELECT symbol, sentiment_score, bullish_count, bearish_count, total_articles, timestamp
                FROM cryptopanic_sentiment
                WHERE timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp
            """)
            sentiment_df = pd.read_sql(sentiment_query, conn, params={
                'start_time': start_time,
                'end_time': end_time
            })
            
            # AlphaVantage health data
            health_query = text("""
                SELECT symbol, health_score, rsi, macd, macd_signal, timestamp
                FROM alphavantage_health
                WHERE timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp
            """)
            health_df = pd.read_sql(health_query, conn, params={
                'start_time': start_time,
                'end_time': end_time
            })
            
            # CoinGecko metrics data
            metrics_query = text("""
                SELECT symbol, market_cap, total_volume, price_change_24h,
                       market_cap_rank, community_score, public_interest_score, timestamp
                FROM coingecko_metrics
                WHERE timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp
            """)
            metrics_df = pd.read_sql(metrics_query, conn, params={
                'start_time': start_time,
                'end_time': end_time
            })
            
            return {
                'cryptopanic': sentiment_df,
                'alphavantage': health_df,
                'coingecko': metrics_df
            }
            
    except Exception as e:
        log.error(f"Error fetching market insights data: {e}")
        return {
            'cryptopanic': pd.DataFrame(),
            'alphavantage': pd.DataFrame(),
            'coingecko': pd.DataFrame()
        }

def calculate_market_stats(insights_data: dict) -> dict:
    """
    Calculates aggregate statistics from market insights data.
    
    Args:
        insights_data: Dictionary of DataFrames from get_market_insights_data()
    
    Returns:
        Dictionary of calculated statistics
    """
    stats = {}
    
    try:
        # CryptoPanic stats
        if not insights_data['cryptopanic'].empty:
            sentiment_df = insights_data['cryptopanic']
            stats['avg_sentiment'] = sentiment_df['sentiment_score'].mean()
            stats['bullish_ratio'] = (
                sentiment_df['bullish_count'].sum() / 
                max(sentiment_df['total_articles'].sum(), 1)
            )
        
        # AlphaVantage stats
        if not insights_data['alphavantage'].empty:
            health_df = insights_data['alphavantage']
            stats['avg_health'] = health_df['health_score'].mean()
            stats['avg_rsi'] = health_df['rsi'].mean()
            stats['macd_signal'] = (
                'Bullish' if (health_df['macd'] > health_df['macd_signal']).mean() > 0.5
                else 'Bearish'
            )
        
        # CoinGecko stats
        if not insights_data['coingecko'].empty:
            metrics_df = insights_data['coingecko']
            stats['total_market_cap'] = metrics_df['market_cap'].sum()
            stats['avg_price_change'] = metrics_df['price_change_24h'].mean()
            stats['avg_community_score'] = metrics_df['community_score'].mean()
            stats['avg_interest_score'] = metrics_df['public_interest_score'].mean()
    
    except Exception as e:
        log.error(f"Error calculating market stats: {e}")
    
    return stats


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')
    print("--- Testing Dashboard Data Provider ---")

    # Ensure DB is available for testing
    if not db_utils.engine:
         print("DB Engine not configured. Cannot run tests.")
    else:
        print("\nFetching overview data...")
        overview = get_overview_data()
        print(f"Overview Keys: {overview.keys()}")
        print(f"Latest Trades DF Shape: {overview.get('latest_trades', pd.DataFrame()).shape}")
        print(f"Disabled Symbols: {overview.get('disabled_symbols')}")

        print("\nFetching market data for BTCUSDT (1h)...")
        market = get_market_data('BTCUSDT', '1h')
        print(f"Market DF Shape: {market.shape}")
        if not market.empty: print(market.head())

        print("\nFetching sentiment data (4h)...")
        sentiment = get_sentiment_data('4h')
        print(f"Sentiment DF Shape: {sentiment.shape}")
        if not sentiment.empty: print(sentiment.head())

        print("\nFetching trading data...")
        open_pos, history = get_trading_data()
        print(f"Open Positions DF Shape: {open_pos.shape}")
        print(f"Trade History DF Shape: {history.shape}")
        if not history.empty: print(history.head())

        print("\nFetching performance data...")
        perf = get_performance_data()
        print(f"Performance DF Shape: {perf.shape}")
        if not perf.empty: print(perf.head())

        print("\nFetching config settings...")
        settings = get_config_settings()
        print(settings)

        print("\nFetching target symbols list...")
        symbols_list = get_target_symbols_list()
        print(symbols_list)

        print("\nFetching market insights data (24h)...")
        insights = get_market_insights_data('24h')
        for key, df in insights.items():
            print(f"{key.capitalize()} DataFrame Shape: {df.shape}")
            if not df.empty: print(df.head())

        print("\nCalculating market stats...")
        stats = calculate_market_stats(insights)
        print(stats)

    print("\n--- Test Complete ---")

