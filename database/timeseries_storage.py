"""
Database storage optimization utilities for cryptocurrency data.
Provides partitioning, aggregation, and efficient data management for time series data.
"""

import logging
import datetime
import pandas as pd
import pytz
from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import asyncpg

# Configure logging
logger = logging.getLogger(__name__)

class TimeSeriesPartitioner:
    """
    Manages partitioning of time-series data to improve database performance
    by creating separate tables for different time periods.
    """
    
    def __init__(self, pool, base_table: str, partition_interval: str = 'monthly'):
        """
        Initialize the partitioner.
        
        Args:
            pool: PostgreSQL connection pool
            base_table: Name of the base table to partition
            partition_interval: 'daily', 'weekly', 'monthly', or 'yearly'
        """
        self.pool = pool
        self.base_table = base_table
        self.partition_interval = partition_interval
        
    async def setup_partitioning(self) -> bool:
        """
        Set up the partitioning schema in the database.
        
        Returns:
            Success status
        """
        try:
            async with self.pool.acquire() as conn:
                # Check if the parent table exists and if it's already partitioned
                table_exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                    self.base_table
                )
                
                # If table exists, check if it's already partitioned
                if table_exists:
                    is_partitioned = await conn.fetchval(
                        "SELECT EXISTS (SELECT FROM pg_partitioned_table pt JOIN pg_class c ON pt.partrelid = c.oid WHERE c.relname = $1)",
                        self.base_table
                    )
                    
                    if is_partitioned:
                        logger.info(f"Table {self.base_table} already exists and is partitioned")
                        return True
                    else:
                        # Table exists but is not partitioned - need to recreate it with partitioning
                        # This is potentially destructive, so we'll create a backup first
                        logger.warning(f"Table {self.base_table} exists but is not partitioned. Creating backup and recreating...")
                        
                        # Create backup of existing table
                        backup_table = f"{self.base_table}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        await conn.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {self.base_table}")
                        logger.info(f"Created backup of existing data in {backup_table}")
                        
                        # Drop existing table
                        await conn.execute(f"DROP TABLE {self.base_table}")
                        logger.info(f"Dropped existing table {self.base_table}")
                        
                        # Create new partitioned table
                        if self.base_table == 'price_data':
                            await self._create_price_data_parent_table(conn)
                        elif self.base_table == 'api_data':
                            await self._create_api_data_parent_table(conn)
                        else:
                            logger.error(f"Unknown base table type: {self.base_table}")
                            return False
                        
                        # Restore data from backup if needed
                        # Note: Uncomment this if you want to automatically restore the data
                        # await conn.execute(f"INSERT INTO {self.base_table} SELECT * FROM {backup_table}")
                        logger.info(f"Created new partitioned table {self.base_table}")
                        logger.info(f"NOTE: Data from {backup_table} was NOT automatically restored to prevent errors")
                        logger.info(f"      Consider manually migrating data if needed")
                else:
                    # Table doesn't exist, create it
                    logger.warning(f"Parent table {self.base_table} does not exist, creating it...")
                    if self.base_table == 'price_data':
                        await self._create_price_data_parent_table(conn)
                    elif self.base_table == 'api_data':
                        await self._create_api_data_parent_table(conn)
                    else:
                        logger.error(f"Unknown base table type: {self.base_table}")
                        return False
                
                logger.info(f"Partitioning setup completed for {self.base_table}")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up partitioning for {self.base_table}: {e}")
            return False
            
    async def _create_price_data_parent_table(self, conn) -> None:
        """Create the price_data parent table with partitioning"""
        await conn.execute("""
            CREATE TABLE price_data (
                id SERIAL,
                symbol VARCHAR(20) NOT NULL,
                interval VARCHAR(10) NOT NULL,
                open_time TIMESTAMP WITH TIME ZONE NOT NULL,
                open NUMERIC(20, 8) NOT NULL,
                high NUMERIC(20, 8) NOT NULL,
                low NUMERIC(20, 8) NOT NULL,
                close NUMERIC(20, 8) NOT NULL,
                volume NUMERIC(30, 8) NOT NULL,
                close_time TIMESTAMP WITH TIME ZONE NOT NULL,
                sma_fast NUMERIC(20, 8),
                sma_slow NUMERIC(20, 8),
                ema_fast NUMERIC(20, 8),
                ema_slow NUMERIC(20, 8),
                rsi_value NUMERIC(20, 8),
                macd_line NUMERIC(20, 8),
                macd_signal NUMERIC(20, 8),
                macd_hist NUMERIC(20, 8),
                fetched_at TIMESTAMP WITH TIME ZONE NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            ) PARTITION BY RANGE (open_time);
            
            -- Create indexes on the parent table (will be inherited by partitions)
            CREATE INDEX idx_price_data_symbol ON price_data(symbol);
            CREATE INDEX idx_price_data_time ON price_data(open_time);
            CREATE INDEX idx_price_data_symbol_time ON price_data(symbol, open_time);
        """)
            
    async def _create_api_data_parent_table(self, conn) -> None:
        """Create the api_data parent table with partitioning"""
        await conn.execute("""
            CREATE TABLE api_data (
                id SERIAL,
                symbol VARCHAR(20) NOT NULL,
                source VARCHAR(50) NOT NULL,
                content_id VARCHAR(255),
                title TEXT,
                url TEXT,
                published_at TIMESTAMP WITH TIME ZONE,
                raw_sentiment_label VARCHAR(50),
                calculated_sentiment_score NUMERIC(10, 6),
                data BYTEA,
                processed_at TIMESTAMP WITH TIME ZONE NOT NULL,
                PRIMARY KEY (symbol, source, processed_at)
            ) PARTITION BY RANGE (processed_at);
            
            -- Create indexes on the parent table
            CREATE INDEX idx_api_data_symbol ON api_data(symbol);
            CREATE INDEX idx_api_data_source ON api_data(source);
            CREATE INDEX idx_api_data_time ON api_data(processed_at);
            CREATE INDEX idx_api_data_symbol_source ON api_data(symbol, source);
        """)
            
    async def create_partition_for_date(self, date: datetime.datetime) -> str:
        """
        Create a partition for a specific date if it doesn't exist.
        
        Args:
            date: The date to create a partition for
            
        Returns:
            Name of the partition table
        """
        # Determine partition name and range based on interval
        partition_name, start_date, end_date = self._get_partition_info(date)
        
        try:
            async with self.pool.acquire() as conn:
                # Check if partition exists
                partition_exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                    partition_name
                )
                
                if not partition_exists:
                    logger.info(f"Creating partition {partition_name} for {start_date} to {end_date}")
                    
                    # Format dates for SQL
                    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S%z")
                    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S%z")
                    
                    # Different column name based on table type
                    time_column = 'open_time' if self.base_table == 'price_data' else 'processed_at'
                    
                    # Create the partition
                    await conn.execute(f"""
                        CREATE TABLE {partition_name} 
                        PARTITION OF {self.base_table} 
                        FOR VALUES FROM ('{start_date_str}') TO ('{end_date_str}')
                    """)
                    
                    logger.info(f"Partition {partition_name} created successfully")
                else:
                    logger.debug(f"Partition {partition_name} already exists")
                    
                return partition_name
                
        except Exception as e:
            logger.error(f"Error creating partition for {date}: {e}")
            return ""
            
    def _get_partition_info(self, date: datetime.datetime) -> Tuple[str, datetime.datetime, datetime.datetime]:
        """
        Get partition name and date range based on the partition interval.
        
        Args:
            date: The date to determine partition info for
            
        Returns:
            Tuple of (partition_name, start_date, end_date)
        """
        date = date.replace(tzinfo=pytz.UTC)
        
        if self.partition_interval == 'daily':
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + datetime.timedelta(days=1)
            partition_name = f"{self.base_table}_{date.strftime('%Y_%m_%d')}"
            
        elif self.partition_interval == 'weekly':
            # Start of week (Monday)
            start_date = (date - datetime.timedelta(days=date.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + datetime.timedelta(days=7)
            partition_name = f"{self.base_table}_{start_date.strftime('%Y_%m_%d')}"
            
        elif self.partition_interval == 'monthly':
            start_date = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if date.month == 12:
                end_date = datetime.datetime(date.year + 1, 1, 1, tzinfo=pytz.UTC)
            else:
                end_date = datetime.datetime(date.year, date.month + 1, 1, tzinfo=pytz.UTC)
            partition_name = f"{self.base_table}_{date.strftime('%Y_%m')}"
            
        else:  # yearly
            start_date = date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.datetime(date.year + 1, 1, 1, tzinfo=pytz.UTC)
            partition_name = f"{self.base_table}_{date.strftime('%Y')}"
            
        return partition_name, start_date, end_date

class TimeSeriesAggregator:
    """
    Aggregates time-series data to reduce storage requirements and improve query performance.
    Creates aggregated tables for different time intervals (hourly, daily, weekly, monthly).
    """
    
    def __init__(self, pool):
        """
        Initialize the aggregator.
        
        Args:
            pool: PostgreSQL connection pool
        """
        self.pool = pool
        
    async def setup_aggregation_tables(self) -> bool:
        """
        Set up the aggregation tables in the database.
        
        Returns:
            Success status
        """
        try:
            async with self.pool.acquire() as conn:
                # Create tables for aggregated price data
                await conn.execute("""
                    -- Create hourly price aggregation table if not exists
                    CREATE TABLE IF NOT EXISTS price_hourly_aggregated (
                        symbol VARCHAR(20) NOT NULL,
                        interval VARCHAR(10) NOT NULL,
                        hour_start TIMESTAMP WITH TIME ZONE NOT NULL,
                        open NUMERIC(20, 8) NOT NULL,
                        high NUMERIC(20, 8) NOT NULL,
                        low NUMERIC(20, 8) NOT NULL,
                        close NUMERIC(20, 8) NOT NULL,
                        volume NUMERIC(30, 8) NOT NULL,
                        candle_count INTEGER NOT NULL,
                        PRIMARY KEY (symbol, interval, hour_start)
                    );
                    
                    -- Create daily price aggregation table if not exists
                    CREATE TABLE IF NOT EXISTS price_daily_aggregated (
                        symbol VARCHAR(20) NOT NULL,
                        interval VARCHAR(10) NOT NULL,
                        day_start TIMESTAMP WITH TIME ZONE NOT NULL,
                        open NUMERIC(20, 8) NOT NULL,
                        high NUMERIC(20, 8) NOT NULL,
                        low NUMERIC(20, 8) NOT NULL,
                        close NUMERIC(20, 8) NOT NULL,
                        volume NUMERIC(30, 8) NOT NULL,
                        candle_count INTEGER NOT NULL,
                        PRIMARY KEY (symbol, interval, day_start)
                    );
                    
                    -- Create weekly price aggregation table if not exists
                    CREATE TABLE IF NOT EXISTS price_weekly_aggregated (
                        symbol VARCHAR(20) NOT NULL,
                        interval VARCHAR(10) NOT NULL,
                        week_start TIMESTAMP WITH TIME ZONE NOT NULL,
                        open NUMERIC(20, 8) NOT NULL,
                        high NUMERIC(20, 8) NOT NULL,
                        low NUMERIC(20, 8) NOT NULL,
                        close NUMERIC(20, 8) NOT NULL,
                        volume NUMERIC(30, 8) NOT NULL,
                        candle_count INTEGER NOT NULL,
                        PRIMARY KEY (symbol, interval, week_start)
                    );
                    
                    -- Create indexes for better query performance
                    CREATE INDEX IF NOT EXISTS idx_hourly_symbol_time ON price_hourly_aggregated(symbol, hour_start);
                    CREATE INDEX IF NOT EXISTS idx_daily_symbol_time ON price_daily_aggregated(symbol, day_start);
                    CREATE INDEX IF NOT EXISTS idx_weekly_symbol_time ON price_weekly_aggregated(symbol, week_start);
                """)
                
                # Create stored procedures for aggregation
                await conn.execute("""
                    -- Create function to aggregate price data by hour
                    CREATE OR REPLACE FUNCTION aggregate_price_data_hourly()
                    RETURNS INTEGER AS $$
                    DECLARE
                        inserted_count INTEGER;
                    BEGIN
                        INSERT INTO price_hourly_aggregated (
                            symbol, 
                            interval, 
                            hour_start, 
                            open, 
                            high, 
                            low, 
                            close, 
                            volume, 
                            candle_count
                        )
                        SELECT 
                            symbol,
                            interval,
                            date_trunc('hour', open_time) as hour_start,
                            FIRST_VALUE(open) OVER (PARTITION BY symbol, interval, date_trunc('hour', open_time) ORDER BY open_time) as open,
                            MAX(high) as high,
                            MIN(low) as low,
                            LAST_VALUE(close) OVER (PARTITION BY symbol, interval, date_trunc('hour', open_time) ORDER BY open_time RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as close,
                            SUM(volume) as volume,
                            COUNT(*) as candle_count
                        FROM price_data
                        WHERE open_time > (SELECT COALESCE(MAX(hour_start), '1970-01-01'::timestamp with time zone) FROM price_hourly_aggregated)
                        GROUP BY symbol, interval, date_trunc('hour', open_time)
                        ON CONFLICT (symbol, interval, hour_start) DO UPDATE 
                        SET 
                            high = GREATEST(price_hourly_aggregated.high, EXCLUDED.high),
                            low = LEAST(price_hourly_aggregated.low, EXCLUDED.low),
                            close = EXCLUDED.close,
                            volume = price_hourly_aggregated.volume + EXCLUDED.volume,
                            candle_count = price_hourly_aggregated.candle_count + EXCLUDED.candle_count;
                            
                        GET DIAGNOSTICS inserted_count = ROW_COUNT;
                        RETURN inserted_count;
                    END;
                    $$ LANGUAGE plpgsql;
                    
                    -- Create function to aggregate price data by day
                    CREATE OR REPLACE FUNCTION aggregate_price_data_daily()
                    RETURNS INTEGER AS $$
                    DECLARE
                        inserted_count INTEGER;
                    BEGIN
                        INSERT INTO price_daily_aggregated (
                            symbol, 
                            interval, 
                            day_start, 
                            open, 
                            high, 
                            low, 
                            close, 
                            volume, 
                            candle_count
                        )
                        SELECT 
                            symbol,
                            interval,
                            date_trunc('day', hour_start) as day_start,
                            FIRST_VALUE(open) OVER (PARTITION BY symbol, interval, date_trunc('day', hour_start) ORDER BY hour_start) as open,
                            MAX(high) as high,
                            MIN(low) as low,
                            LAST_VALUE(close) OVER (PARTITION BY symbol, interval, date_trunc('day', hour_start) ORDER BY hour_start RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as close,
                            SUM(volume) as volume,
                            SUM(candle_count) as candle_count
                        FROM price_hourly_aggregated
                        WHERE hour_start > (SELECT COALESCE(MAX(day_start), '1970-01-01'::timestamp with time zone) FROM price_daily_aggregated)
                        GROUP BY symbol, interval, date_trunc('day', hour_start)
                        ON CONFLICT (symbol, interval, day_start) DO UPDATE 
                        SET 
                            high = GREATEST(price_daily_aggregated.high, EXCLUDED.high),
                            low = LEAST(price_daily_aggregated.low, EXCLUDED.low),
                            close = EXCLUDED.close,
                            volume = price_daily_aggregated.volume + EXCLUDED.volume,
                            candle_count = price_daily_aggregated.candle_count + EXCLUDED.candle_count;
                            
                        GET DIAGNOSTICS inserted_count = ROW_COUNT;
                        RETURN inserted_count;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                
                logger.info("Aggregation tables and functions set up successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up aggregation tables: {e}")
            return False
            
    async def aggregate_data(self, force_full: bool = False) -> Dict[str, int]:
        """
        Run the aggregation process to transform raw data into aggregated tables.
        
        Args:
            force_full: Whether to force a full aggregation instead of incremental
            
        Returns:
            Dictionary with counts of aggregated rows
        """
        results = {
            'hourly': 0,
            'daily': 0,
            'weekly': 0
        }
        
        try:
            async with self.pool.acquire() as conn:
                # Check if the price_data table exists and has data
                price_data_exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'price_data')"
                )
                
                if not price_data_exists:
                    logger.warning("Cannot aggregate data: price_data table does not exist")
                    return results
                    
                # Check if there's any data to aggregate
                data_count = await conn.fetchval("SELECT COUNT(*) FROM price_data LIMIT 1")
                if data_count == 0:
                    logger.info("No data in price_data table to aggregate")
                    return results
                    
                try:
                    # Aggregate hourly data
                    hourly_count = await conn.fetchval("SELECT aggregate_price_data_hourly()")
                    results['hourly'] = hourly_count or 0
                    logger.info(f"Aggregated {hourly_count} hourly price records")
                except Exception as e:
                    logger.error(f"Error during hourly aggregation: {e}")
                    # Continue with daily aggregation even if hourly fails
                    
                try:
                    # Check if hourly aggregated table has data before proceeding with daily
                    hourly_data_exists = await conn.fetchval(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'price_hourly_aggregated')"
                    )
                    
                    if hourly_data_exists:
                        hourly_count = await conn.fetchval("SELECT COUNT(*) FROM price_hourly_aggregated LIMIT 1")
                        if hourly_count > 0:
                            # Aggregate daily data
                            daily_count = await conn.fetchval("SELECT aggregate_price_data_daily()")
                            results['daily'] = daily_count or 0
                            logger.info(f"Aggregated {daily_count} daily price records")
                        else:
                            logger.info("No hourly data to aggregate into daily records")
                    else:
                        logger.warning("Cannot aggregate daily data: price_hourly_aggregated table does not exist")
                except Exception as e:
                    logger.error(f"Error during daily aggregation: {e}")
                    
                return results
                
        except Exception as e:
            logger.error(f"Error during data aggregation: {e}")
            return results
            
    async def get_price_data(self, 
                           symbol: str, 
                           start_time: datetime.datetime, 
                           end_time: datetime.datetime, 
                           interval: str = 'auto') -> pd.DataFrame:
        """
        Get price data with automatic resolution selection based on time range.
        
        Args:
            symbol: The cryptocurrency symbol
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            interval: Specific interval or 'auto' to select based on range
            
        Returns:
            DataFrame with the requested price data
        """
        # Determine appropriate resolution based on time range
        if interval == 'auto':
            time_range_days = (end_time - start_time).total_seconds() / 86400
            
            if time_range_days <= 3:  # Less than 3 days, use raw data
                table = 'price_data'
                time_col = 'open_time'
            elif time_range_days <= 30:  # Less than a month, use hourly data
                table = 'price_hourly_aggregated'
                time_col = 'hour_start'
            elif time_range_days <= 180:  # Less than 6 months, use daily data
                table = 'price_daily_aggregated'
                time_col = 'day_start'
            else:  # More than 6 months, use weekly data
                table = 'price_weekly_aggregated'
                time_col = 'week_start'
        else:
            # Use the specified interval
            table = f"price_{interval}_aggregated" if interval != 'raw' else 'price_data'
            time_col = f"{interval}_start" if interval != 'raw' else 'open_time'
            
        try:
            async with self.pool.acquire() as conn:
                # Query the appropriate table
                query = f"""
                    SELECT * FROM {table}
                    WHERE symbol = $1 AND {time_col} BETWEEN $2 AND $3
                    ORDER BY {time_col} ASC
                """
                
                rows = await conn.fetch(query, symbol, start_time, end_time)
                
                # Convert to DataFrame
                if rows:
                    df = pd.DataFrame(rows, columns=rows[0].keys())
                    logger.info(f"Retrieved {len(df)} {interval if interval != 'auto' else table.split('_')[1]} price records for {symbol}")
                    return df
                else:
                    logger.warning(f"No price data found for {symbol} in requested time range")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error retrieving price data: {e}")
            return pd.DataFrame()