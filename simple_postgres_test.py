#!/usr/bin/env python3
"""
Simple PostgreSQL Database Test Script

Tests basic database connectivity with PostgreSQL and creates tables if they don't exist.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
import asyncpg
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Get database URL from environment or .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    log.warning("python-dotenv not installed, assuming DATABASE_URL is in environment.")

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    log.error("DATABASE_URL not found in environment or .env file.")
    sys.exit(1)

log.info(f"Using database URL: {DATABASE_URL[:DATABASE_URL.index('@') if '@' in DATABASE_URL else 10]}...") 

async def test_asyncpg_connection():
    """Test database connection using asyncpg (async PostgreSQL driver)."""
    log.info("Testing asyncpg connection...")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        log.info("Connected successfully using asyncpg!")
        
        # Check server version
        version = await conn.fetchval("SELECT version()")
        log.info(f"PostgreSQL version: {version}")
        
        # List tables
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        if tables:
            log.info(f"Found {len(tables)} tables in database:")
            for table in tables:
                log.info(f"  - {table['table_name']}")
        else:
            log.info("No tables found in database. You may need to run the table creation script.")
        
        await conn.close()
        return True
    except Exception as e:
        log.error(f"asyncpg connection failed: {e}")
        return False

def test_sqlalchemy_connection():
    """Test database connection using SQLAlchemy (sync PostgreSQL driver)."""
    log.info("Testing SQLAlchemy connection...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            log.info("Connected successfully using SQLAlchemy!")
            
            # Check server version
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            log.info(f"PostgreSQL version: {version}")
            
            # List tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            
            tables = result.fetchall()
            if tables:
                log.info(f"Found {len(tables)} tables in database:")
                for table in tables:
                    log.info(f"  - {table[0]}")
            else:
                log.info("No tables found in database. You may need to run the table creation script.")
            
        return True
    except Exception as e:
        log.error(f"SQLAlchemy connection failed: {e}")
        return False

async def main():
    log.info("=== Testing PostgreSQL Database Connection ===")
    
    # Test SQLAlchemy (synchronous) connection
    sqlalchemy_success = test_sqlalchemy_connection()
    
    # Test asyncpg (asynchronous) connection
    asyncpg_success = await test_asyncpg_connection()
    
    if sqlalchemy_success and asyncpg_success:
        log.info("\n✅ All database connection tests PASSED")
        return 0
    elif sqlalchemy_success:
        log.warning("\n⚠️ SQLAlchemy test PASSED but asyncpg test FAILED")
        return 1
    elif asyncpg_success:
        log.warning("\n⚠️ asyncpg test PASSED but SQLAlchemy test FAILED")
        return 1
    else:
        log.error("\n❌ All database connection tests FAILED")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)