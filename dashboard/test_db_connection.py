"""
Test script to verify database connectivity for the dashboard
"""
import asyncio
import logging
import sys
import os

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import config
from database.db_utils import get_pool, async_fetch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def test_db_connection():
    """Test database connectivity"""
    log.info("Testing database connection...")
    try:
        # Test connection pool
        pool = await get_pool()
        if not pool:
            log.error("Failed to create database connection pool")
            return False
        
        # Test a simple query
        result = await async_fetch("SELECT 1 as test")
        if result and result[0]['test'] == 1:
            log.info("Database connection successful!")
            return True
        else:
            log.error("Database query failed")
            return False
    except Exception as e:
        log.error(f"Database connection error: {e}")
        return False

async def test_get_price_data():
    """Test price data retrieval"""
    log.info("Testing price data retrieval...")
    try:
        pool = await get_pool()
        if not pool:
            return False
        
        # Check if price_data table exists and has data
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'price_data'
        )
        """
        result = await async_fetch(query)
        if not result[0]['exists']:
            log.warning("price_data table does not exist")
            return False
        
        # Try to get some data
        query = """
        SELECT COUNT(*) FROM price_data
        """
        result = await async_fetch(query)
        count = result[0]['count']
        log.info(f"Found {count} price data records")
        
        if count > 0:
            # Get a sample
            query = """
            SELECT * FROM price_data LIMIT 1
            """
            data = await async_fetch(query)
            log.info(f"Sample data: {data[0]}")
            return True
        else:
            log.warning("No price data found")
            return False
    except Exception as e:
        log.error(f"Error testing price data: {e}")
        return False

async def main():
    """Run database connectivity tests"""
    log.info(f"Database URL: {config.DATABASE_URL.replace(':'+config.DATABASE_URL.split(':')[-1], ':*****')}")
    
    conn_ok = await test_db_connection()
    if not conn_ok:
        log.error("Basic database connectivity test failed")
        return
    
    data_ok = await test_get_price_data()
    if not data_ok:
        log.warning("Price data test did not find data")
    
    log.info("Database tests completed")

if __name__ == "__main__":
    asyncio.run(main())