# /database/db_utils.py

"""
Database Utility Module

This module provides utility functions and table definitions for interacting with the database.
Now optimized for PostgreSQL with proper async support.

Functions:
    init_db: Initializes database tables if they don't exist.
    get_db_session: Provides a context manager for database sessions.
    bulk_insert_data: Inserts data into a table with optional duplicate checks.
    df_to_db: Writes a Pandas DataFrame to a database table.
    set_config_value: Stores a configuration value in the database.
    get_config_value: Retrieves a configuration value from the database.

Table Definitions:
    price_data: Stores price data with technical indicators.
    news_data: Stores news articles and metadata.
    reddit_data: Stores Reddit posts and metadata.
    twitter_data: Stores Twitter data and metadata.
    sentiment_analysis_results: Stores sentiment analysis results.
    trade_log: Stores trade logs and metadata.
    cryptopanic_sentiment: Stores sentiment data from CryptoPanic API.
    alphavantage_health: Stores health metrics from AlphaVantage API.
    coingecko_metrics: Stores market metrics from CoinGecko API.
    app_config: Stores application configuration key-value pairs.
"""

import logging
import pandas as pd
import numpy as np # Added for replacing inf/-inf
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, JSON, UniqueConstraint, Index, inspect, BigInteger, select
from sqlalchemy.dialects.postgresql import JSONB # PostgreSQL JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from contextlib import contextmanager
import math # Added for ceil function
from typing import List, Dict, Any, Optional
from typing_extensions import Literal
import asyncpg
import datetime
import json

# Assuming config.py is in the parent directory or project root added to PYTHONPATH
import config # Use absolute import if project structure allows

log = logging.getLogger(__name__)

# --- Database Setup ---
DATABASE_URL = config.DATABASE_URL
engine = None
SessionLocal = None
metadata = MetaData()

if DATABASE_URL:
    try:
        # Check if using PostgreSQL
        is_postgresql = 'postgresql' in DATABASE_URL or 'postgres' in DATABASE_URL
        
        connect_args = {}
        engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, connect_args=connect_args)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        log.info(f"Database engine created for URL: {'...' + DATABASE_URL[-50:]}") # Log partial URL for security
    except Exception as e:
        log.error(f"Failed to create database engine or session maker: {e}", exc_info=True)
        # Application might need to exit or run in a limited mode
else:
    log.critical("DATABASE_URL not configured. Database features will be unavailable.")
    # Application might need to exit or run in a limited mode


# --- Define Database Tables ---
# Use JSONB for PostgreSQL
JSON_TYPE = JSONB if engine and engine.dialect.name == 'postgresql' else JSON

price_data = Table(
    'price_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(32), nullable=False),
    Column('interval', String(8), nullable=False), # e.g., '5m'
    Column('open_time', DateTime(timezone=True), nullable=False),
    Column('open', Float, nullable=False),
    Column('high', Float, nullable=False),
    Column('low', Float, nullable=False),
    Column('close', Float, nullable=False),
    Column('volume', Float, nullable=False),
    Column('close_time', DateTime(timezone=True), nullable=False),
    # Technical Indicators - Adjust names based on actual calculation output
    Column('sma_fast', Float, nullable=True),
    Column('sma_slow', Float, nullable=True),
    Column('ema_fast', Float, nullable=True),
    Column('ema_slow', Float, nullable=True),
    Column('rsi_value', Float, nullable=True),
    Column('macd_line', Float, nullable=True),
    Column('macd_signal', Float, nullable=True),
    Column('macd_hist', Float, nullable=True),
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Constraints and Indexes
    UniqueConstraint('symbol', 'interval', 'open_time', name='uq_price_data_symbol_interval_time'),
    Index('ix_price_data_symbol_interval_open_time', 'symbol', 'interval', 'open_time'),
    Index('ix_price_data_open_time', 'open_time') # Index for time-based queries
)

news_data = Table(
    'news_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('source_api', String(50), nullable=False), # e.g., 'cryptonews'
    Column('source_publisher', String(100), nullable=True), # e.g., 'CoinDesk'
    Column('article_id', String(255), nullable=True), # Unique ID from source API
    Column('title', String(512), nullable=False),
    Column('text_content', Text, nullable=True),
    Column('url', String(1024), nullable=False, unique=True),
    Column('published_at', DateTime(timezone=True), nullable=False), # Stored in UTC
    Column('tickers_mentioned', JSON_TYPE, nullable=True), # Store as list ['BTC', 'ETH']
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes
    Index('ix_news_data_published_at', 'published_at'),
    Index('ix_news_data_source_api', 'source_api')
)

reddit_data = Table(
    'reddit_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('post_id', String(20), nullable=False, unique=True), # Reddit Post ID
    Column('subreddit', String(100), nullable=False),
    Column('title', String(512), nullable=False),
    Column('selftext', Text, nullable=True),
    Column('url', String(1024), nullable=False),
    Column('score', Integer, nullable=True),
    Column('num_comments', Integer, nullable=True),
    Column('created_utc', DateTime(timezone=True), nullable=False), # Timestamp from Reddit
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes
    Index('ix_reddit_data_subreddit_created', 'subreddit', 'created_utc'),
    Index('ix_reddit_data_created_utc', 'created_utc')
)

twitter_data = Table(
    'twitter_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('tweet_id', String(30), nullable=False, unique=True), # Twitter Tweet ID
    Column('author_id', String(30), nullable=True),
    Column('text', Text, nullable=False),
    Column('created_at', DateTime(timezone=True), nullable=False), # Timestamp from Twitter (UTC)
    Column('public_metrics', JSON_TYPE, nullable=True), # Store dict: {'retweet_count': .., 'reply_count': .., 'like_count': .., 'quote_count': ..}
    Column('hashtags', JSON_TYPE, nullable=True), # Store as list
    Column('cashtags', JSON_TYPE, nullable=True), # Store as list
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes
    Index('ix_twitter_data_created_at', 'created_at')
)

sentiment_analysis_results = Table(
    'sentiment_analysis_results', metadata,
    Column('id', Integer, primary_key=True),
    # Foreign keys to link to the source data
    Column('news_id', Integer, nullable=True), # FK to news_data.id
    Column('reddit_id', Integer, nullable=True), # FK to reddit_data.id
    Column('twitter_id', Integer, nullable=True), # FK to twitter_data.id
    Column('model_name', String(100), nullable=False), # Sentiment model used
    Column('sentiment_label', String(20), nullable=False), # e.g., 'positive', 'negative', 'neutral'
    Column('sentiment_score', Float, nullable=False), # e.g., -1.0 to 1.0
    Column('analyzed_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes - Index individual FKs or create composite indexes if needed
    Index('ix_sentiment_news_id', 'news_id'),
    Index('ix_sentiment_reddit_id', 'reddit_id'),
    Index('ix_sentiment_twitter_id', 'twitter_id'),
    Index('ix_sentiment_analyzed_at', 'analyzed_at')
    # TODO: Add constraint to ensure only one FK is non-null if DB supports it easily,
    # otherwise handle in application logic or triggers.
)

trade_log = Table(
    'trade_log', metadata,
    Column('id', Integer, primary_key=True),
    Column('timestamp', DateTime(timezone=True), server_default=func.now()),
    Column('symbol', String(32), nullable=False),
    Column('trade_type', String(4), nullable=False), # 'BUY' or 'SELL'
    Column('order_type', String(10), nullable=False), # 'MARKET', 'LIMIT', etc.
    Column('status', String(20), nullable=False), # 'FILLED', 'FAILED', 'SIMULATED'
    Column('binance_order_id', String(50), nullable=True), # Actual Order ID from Binance
    Column('price', Float, nullable=False), # Average execution price
    Column('quantity', Float, nullable=False),
    Column('usd_value', Float, nullable=False), # Total value of trade
    Column('fee', Float, nullable=True), # Trading fee paid
    Column('pnl', Float, nullable=True), # Realized Profit/Loss for SELL trades
    Column('signal_confidence', Float, nullable=True), # Confidence score from prediction model
    Column('trigger_reason', String(50), nullable=True), # 'Model Signal', 'Stop Loss', etc.
    Column('trading_mode', String(10), nullable=False), # 'PAPER' or 'LIVE'
    # Indexes
    Index('ix_trade_log_symbol_timestamp', 'symbol', 'timestamp'),
    Index('ix_trade_log_timestamp', 'timestamp')
)

cryptopanic_sentiment = Table(
    'cryptopanic_sentiment',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(20), nullable=False),
    Column('sentiment_score', Float, nullable=False),
    Column('bullish_count', Integer),
    Column('bearish_count', Integer),
    Column('total_articles', Integer),
    Column('timestamp', DateTime(timezone=True), nullable=False),
    Index('idx_cryptopanic_symbol_timestamp', 'symbol', 'timestamp')
)

alphavantage_health = Table(
    'alphavantage_health',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(20), nullable=False),
    Column('health_score', Float, nullable=False),
    Column('rsi', Float),
    Column('macd', Float),
    Column('macd_signal', Float),
    Column('timestamp', DateTime(timezone=True), nullable=False),
    Index('idx_alphavantage_symbol_timestamp', 'symbol', 'timestamp')
)

coingecko_metrics = Table(
    'coingecko_metrics',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(20), nullable=False),
    Column('market_cap', BigInteger),
    Column('total_volume', BigInteger),
    Column('price_change_24h', Float),
    Column('market_cap_rank', Integer),
    Column('community_score', Float),
    Column('public_interest_score', Float),
    Column('timestamp', DateTime(timezone=True), nullable=False),
    Index('idx_coingecko_symbol_timestamp', 'symbol', 'timestamp')
)


# --- Database Utility Functions ---

def init_db():
    """Creates database tables if they don't exist."""
    if not engine:
        log.error("Cannot initialize database, engine not configured.")
        return False
    if not metadata.tables:
        log.warning("No tables defined in metadata, skipping init_db.")
        return False
    try:
        log.info("Attempting to create database tables if they don't exist...")
        metadata.create_all(bind=engine)
        log.info("Database tables checked/created successfully.")
        return True
    except Exception as e:
        log.error(f"Unexpected error initializing database: {e}", exc_info=True)
        return False

@contextmanager
def get_db_session():
    """Provides a database session using a context manager pattern."""
    if not SessionLocal:
        log.error("SessionLocal not configured. Cannot get DB session.")
        yield None # Or raise an exception
        return

    db = SessionLocal()
    try:
        log.debug("DB Session opened.")
        yield db
    except SQLAlchemyError as e:
         log.error(f"Database session error: {e}", exc_info=True)
         db.rollback()
         raise # Re-raise the exception after rollback
    except Exception as e:
         log.error(f"Unexpected error in DB session context: {e}", exc_info=True)
         db.rollback()
         raise
    finally:
        log.debug("DB Session closed.")
        db.close()

def bulk_insert_data(data_list: List[Dict[str, Any]], table: Table, unique_column: Optional[str] = None, chunk_size: int = 900) -> None:
    """Bulk insert data with duplicate checking."""
    if not data_list:
        return

    if not engine:
        log.error("Database engine is not configured. Cannot perform bulk insert.")
        return
        
    if unique_column and unique_column in data_list[0]:
        unique_values = [row[unique_column] for row in data_list]
        with engine.connect() as conn:
            # Modern SQLAlchemy select syntax
            stmt = select(table.c[unique_column]).where(table.c[unique_column].in_(unique_values))
            existing = set(row[0] for row in conn.execute(stmt))
            
            # Filter out existing records
            data_list = [row for row in data_list if row[unique_column] not in existing]

    # Process in chunks
    for i in range(0, len(data_list), chunk_size):
        chunk = data_list[i:i + chunk_size]
        if chunk:
            with engine.connect() as conn:
                conn.execute(table.insert(), chunk)
                conn.commit()

def df_to_db(df: pd.DataFrame, table_name: str, if_exists: Literal['fail', 'replace', 'append'] = 'append', index: bool = False) -> None:
    """Write DataFrame to SQL with timezone handling.
    
    Args:
        df: Pandas DataFrame to write to the database
        table_name: Name of the table to write to
        if_exists: What to do if the table exists ('fail', 'replace', or 'append')
        index: Whether to write DataFrame index as a column
    """
    if df.empty:
        log.warning(f"Empty DataFrame provided to df_to_db for table {table_name}")
        return
        
    df = df.copy()
    
    # Convert timezone-aware datetime columns to UTC with robust type checking
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_cols:
        try:
            # More robust check for timezone attributes with better type safety
            if hasattr(df[col], 'dt'):
                dt_accessor = df[col].dt
                
                # Check if it has timezone info with proper attribute checking
                has_tz = False
                if hasattr(dt_accessor, 'tz'):
                    has_tz = dt_accessor.tz is not None
                
                # Only convert if it has a timezone and the convert method exists
                if has_tz and hasattr(dt_accessor, 'tz_convert'):
                    df[col] = dt_accessor.tz_convert('UTC')
        except (AttributeError, TypeError, ValueError) as e:
            log.warning(f"Error converting timezone for column {col}: {e}")
    
    # Replace inf/-inf values with None (SQL doesn't handle infinities)
    df = df.replace([np.inf, -np.inf], None)
    
    if engine:
        try:
            # No need for validation - type checking is handled by the Literal type
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists=if_exists,
                index=index,
                chunksize=1000  # Process in chunks for better memory usage
            )
            log.debug(f"Successfully wrote {len(df)} rows to table {table_name}")
        except Exception as e:
            log.error(f"Error writing DataFrame to database: {e}")
    else:
        log.error("Database engine not configured. Cannot write DataFrame to database.")

def set_config_value(key: str, value: Any) -> bool:
    """
    Stores a configuration value in the database.
    
    Args:
        key: The configuration key
        value: The configuration value to store
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not engine:
            log.error("Database engine is not configured. Cannot store configuration.")
            return False
            
        # Create config table if it doesn't exist
        if not _ensure_config_table():
            return False
            
        # Serialize complex values to JSON if needed
        value_to_store = value
        if not isinstance(value, (str, int, float, bool, type(None))):
            value_to_store = json.dumps(value)
            
        with engine.begin() as conn:
            # Check if key exists
            stmt = select(config_table).where(config_table.c.key == key)
            result = conn.execute(stmt).fetchone()
            
            if result:
                # Update existing key
                update_stmt = config_table.update().where(config_table.c.key == key).values(
                    value=value_to_store,
                    updated_at=func.now()
                )
                conn.execute(update_stmt)
            else:
                # Insert new key
                insert_stmt = config_table.insert().values(
                    key=key,
                    value=value_to_store,
                    created_at=func.now(),
                    updated_at=func.now()
                )
                conn.execute(insert_stmt)
                
        log.debug(f"Configuration value set successfully: {key}")
        return True
    except Exception as e:
        log.error(f"Error setting configuration value: {e}")
        return False

def _ensure_config_table() -> bool:
    """
    Ensures the config table exists in the database.
    
    Returns:
        True if the table exists or was created successfully, False otherwise
    """
    try:
        if not engine:
            log.error("Database engine is not configured. Cannot create config table.")
            return False
            
        # Define the config table if not already defined
        global config_table
        if 'config_table' not in globals():
            config_table = Table(
                'app_config',
                metadata,
                Column('id', Integer, primary_key=True),
                Column('key', String(255), nullable=False, unique=True, index=True),
                Column('value', Text, nullable=True),
                Column('created_at', DateTime(timezone=True), server_default=func.now()),
                Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
            )
            
        # Create the table if it doesn't exist
        if not inspect(engine).has_table(config_table.name):
            config_table.create(engine)
            log.info(f"Created configuration table: {config_table.name}")
            
        return True
    except Exception as e:
        log.error(f"Error ensuring config table: {e}")
        return False

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Retrieves a configuration value from the database.
    
    Args:
        key: The configuration key to retrieve
        default: The default value to return if the key is not found
        
    Returns:
        The configuration value, or the default if not found
    """
    try:
        if not engine:
            log.error("Database engine is not configured. Cannot retrieve configuration.")
            return default
            
        # Ensure the config table exists
        if not _ensure_config_table():
            return default
            
        with engine.connect() as conn:
            stmt = select(config_table.c.value).where(config_table.c.key == key)
            result = conn.execute(stmt).fetchone()
            
            if result and result[0] is not None:
                value = result[0]
                
                # Try to parse JSON for complex values
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, return as is
                        return value
                return value
            return default
    except Exception as e:
        log.error(f"Error getting configuration value: {e}")
        return default

# --- Async Database Utility Functions ---

async def get_db_pool():
    """
    Get a connection pool to the database.
    
    Note: For SQL Server connections, we'll use aioodbc instead of asyncpg
    as asyncpg only supports PostgreSQL.
    """
    try:
        # Check if we're using PostgreSQL
        if 'postgresql' in config.DATABASE_URL or 'postgres' in config.DATABASE_URL:
            import asyncpg
            return await asyncpg.create_pool(config.DATABASE_URL)
        
        # For SQL Server, use aioodbc (would need to be installed)
        elif 'mssql' in config.DATABASE_URL:
            # Fallback to synchronous operations for SQL Server
            log.warning("Async operations with SQL Server are not supported by asyncpg. Using synchronous operations instead.")
            return None
        else:
            log.error(f"Unsupported database type for async operations: {config.DATABASE_URL.split('://')[0]}")
            return None
    except ImportError:
        log.error("Required package asyncpg or aioodbc not installed")
        return None
    except Exception as e:
        log.error(f"Error creating database pool: {e}")
        return None

async def store_cryptopanic_sentiment(data: Dict[str, Any]) -> bool:
    """Store CryptoPanic sentiment data in the database."""
    try:
        # For SQL Server, use SQLAlchemy instead of asyncpg
        if 'mssql' in config.DATABASE_URL:
            # Extract data
            symbol = data.get('symbol', '')
            results = data.get('results', [])
            timestamp = data.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
            
            # Calculate sentiment metrics
            bullish_count = sum(1 for r in results if r.get('sentiment') == 'bullish')
            bearish_count = sum(1 for r in results if r.get('sentiment') == 'bearish')
            total_articles = len(results)
            
            # Calculate sentiment score
            sentiment_score = 0
            if total_articles > 0:
                sentiment_score = (bullish_count - bearish_count) / total_articles
            
            # Use synchronous SQLAlchemy API
            if engine:
                with engine.connect() as conn:
                    insert_stmt = cryptopanic_sentiment.insert().values(
                        symbol=symbol,
                        sentiment_score=sentiment_score,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        total_articles=total_articles,
                        timestamp=timestamp
                    )
                    conn.execute(insert_stmt)
                    conn.commit()
                    
                log.info(f"Stored CryptoPanic sentiment data for {symbol}")
                return True
            else:
                log.error("Database engine not available")
                return False
        
        # For PostgreSQL, use asyncpg
        else:
            pool = await get_db_pool()
            if not pool:
                log.error("Could not get database pool")
                return False
                
            async with pool.acquire() as conn:
                # Extract sentiment metrics
                symbol = data.get('symbol', '')
                results = data.get('results', [])
                
                # Calculate sentiment score from results
                bullish_count = sum(1 for r in results if r.get('sentiment') == 'bullish')
                bearish_count = sum(1 for r in results if r.get('sentiment') == 'bearish')
                total_articles = len(results)
                
                # Simple sentiment score calculation (-1 to 1 range)
                sentiment_score = 0
                if total_articles > 0:
                    sentiment_score = (bullish_count - bearish_count) / total_articles
                
                # Insert into database with current timestamp
                await conn.execute('''
                    INSERT INTO cryptopanic_sentiment (
                        symbol,
                        sentiment_score,
                        bullish_count,
                        bearish_count,
                        total_articles,
                        timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                ''', 
                    symbol,
                    sentiment_score,
                    bullish_count,
                    bearish_count,
                    total_articles,
                    datetime.datetime.now(datetime.timezone.utc)
                )
                
        return True
    except Exception as e:
        log.error(f"Error storing CryptoPanic data: {e}")
        return False

async def store_alphavantage_health(data: Dict[str, Any]) -> bool:
    """Store AlphaVantage health index data in the database."""
    try:
        # For SQL Server, use SQLAlchemy instead of asyncpg
        if 'mssql' in config.DATABASE_URL:
            # Extract values safely with proper type checking
            symbol = data.get('symbol', '')
            
            # Extract health metrics with proper null-safety
            health_score = float(data.get('health_score', 0))
            rsi = float(data.get('rsi', 0))
            macd = float(data.get('macd', 0))
            macd_signal = float(data.get('macd_signal', 0))
            timestamp = data.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
            
            # Use synchronous SQLAlchemy API
            if engine:
                with engine.connect() as conn:
                    insert_stmt = alphavantage_health.insert().values(
                        symbol=symbol,
                        health_score=health_score,
                        rsi=rsi,
                        macd=macd,
                        macd_signal=macd_signal,
                        timestamp=timestamp
                    )
                    conn.execute(insert_stmt)
                    conn.commit()
                    
                log.info(f"Stored AlphaVantage health data for {symbol}")
                return True
            else:
                log.error("Database engine not available")
                return False
        
        # For PostgreSQL, use asyncpg
        else:
            pool = await get_db_pool()
            if not pool:
                log.error("Could not get database pool")
                return False
                
            async with pool.acquire() as conn:
                # Extract values safely with proper type checking
                symbol = data.get('symbol', '')
                
                # Extract health metrics with proper null-safety
                health_score = float(data.get('health_score', 0))
                rsi = float(data.get('rsi', 0))
                macd = float(data.get('macd', 0))
                macd_signal = float(data.get('macd_signal', 0))
                
                await conn.execute('''
                    INSERT INTO alphavantage_health (
                        symbol,
                        health_score,
                        rsi,
                        macd,
                        macd_signal,
                        timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                ''', 
                    symbol,
                    health_score,
                    rsi,
                    macd,
                    macd_signal,
                    datetime.datetime.now(datetime.timezone.utc)
                )
        
        return True
    except Exception as e:
        log.error(f"Error storing AlphaVantage data: {e}")
        return False

async def store_coingecko_metrics(data: Dict[str, Any]) -> bool:
    """Store CoinGecko metrics data in the database."""
    try:
        # For SQL Server, use SQLAlchemy instead of asyncpg
        if 'mssql' in config.DATABASE_URL:
            # Extract metrics safely
            symbol = data.get('symbol', '').upper()
            market_cap = data.get('market_cap', 0)
            total_volume = data.get('total_volume', 0)
            price_change_24h = data.get('price_change_24h', 0)
            market_cap_rank = data.get('market_cap_rank', 0)
            community_score = data.get('community_score', 0)
            public_interest_score = data.get('public_interest_score', 0)
            timestamp = data.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
            
            # Use synchronous SQLAlchemy API
            if engine:
                with engine.connect() as conn:
                    insert_stmt = coingecko_metrics.insert().values(
                        symbol=symbol,
                        market_cap=market_cap,
                        total_volume=total_volume,
                        price_change_24h=price_change_24h,
                        market_cap_rank=market_cap_rank,
                        community_score=community_score,
                        public_interest_score=public_interest_score,
                        timestamp=timestamp
                    )
                    conn.execute(insert_stmt)
                    conn.commit()
                    
                log.info(f"Stored CoinGecko metrics data for {symbol}")
                return True
            else:
                log.error("Database engine not available")
                return False
        
        # For PostgreSQL, use asyncpg
        else:
            pool = await get_db_pool()
            if not pool:
                log.error("Could not get database pool")
                return False
                
            async with pool.acquire() as conn:
                # Extract metrics with safe get methods to prevent attribute errors
                symbol = data.get('symbol', '').upper()
                
                # Extract nested values safely with default empty dictionaries
                market_cap = data.get('market_cap', 0)
                total_volume = data.get('total_volume', 0)
                price_change_24h = data.get('price_change_24h', 0)
                market_cap_rank = data.get('market_cap_rank', 0)
                community_score = data.get('community_score', 0)
                public_interest_score = data.get('public_interest_score', 0)
                
                await conn.execute('''
                    INSERT INTO coingecko_metrics (
                        symbol,
                        market_cap,
                        total_volume,
                        price_change_24h,
                        market_cap_rank,
                        community_score,
                        public_interest_score,
                        timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''', 
                    symbol,
                    market_cap,
                    total_volume,
                    price_change_24h,
                    market_cap_rank,
                    community_score,
                    public_interest_score,
                    datetime.datetime.now(datetime.timezone.utc)
                )
        
        return True
    except Exception as e:
        log.error(f"Error storing CoinGecko data: {e}")
        return False


# Example of running init_db directly (for setup)
# Run using 'python -m database.db_utils' from the project root directory
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')
    log_main = logging.getLogger(__name__)

    print("Attempting to initialize database schema (Direct Run)...")
    if config.DATABASE_URL:
        if init_db():
             print("Database initialization check complete.")
        else:
             print("Database initialization failed. Check logs.")
    else:
         print("DATABASE_URL not found in config, skipping initialization.")