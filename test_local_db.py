#!/usr/bin/env python3
"""
Test script for local PostgreSQL or SQLite as a fallback
"""

import os
import sys
import logging
import sqlite3
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Try different connection options
def test_local_postgres():
    """Try connecting to a local PostgreSQL instance"""
    log.info("Attempting to connect to local PostgreSQL...")
    try:
        # Try connecting to local PostgreSQL
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            log.info(f"✅ Connected to local PostgreSQL: {version}")
            return True
    except Exception as e:
        log.warning(f"Local PostgreSQL connection failed: {e}")
        return False

def test_sqlite():
    """Try using SQLite as a fallback"""
    log.info("Using SQLite as a fallback database...")
    try:
        # Use SQLite for local development/testing
        engine = create_engine("sqlite:///cyclone_test.db")
        
        # Create schema
        metadata = MetaData()
        test_table = Table(
            'cyclone_test', metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String(50), nullable=False),
            Column('value', String(100), nullable=True)
        )
        
        # Create table
        metadata.create_all(engine)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO cyclone_test (name, value) VALUES ('test', 'SQLite test')"))
            conn.commit()
            
            result = conn.execute(text("SELECT * FROM cyclone_test"))
            rows = result.fetchall()
            log.info(f"SQLite test table has {len(rows)} rows")
            for row in rows:
                log.info(f"  - {row}")
                
        log.info("✅ SQLite database connection and operations successful")
        
        # Update .env file to use SQLite
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                content = f.read()
            
            # Replace PostgreSQL connection with SQLite
            if 'DATABASE_URL=' in content:
                modified_content = content.replace(
                    content.split('DATABASE_URL=')[1].split('\n')[0],
                    "sqlite:///cyclone_test.db"
                )
                
                with open('.env', 'w') as f:
                    f.write(modified_content)
                    
                log.info("Updated .env file to use SQLite")
        
        return True
    except Exception as e:
        log.error(f"SQLite setup failed: {e}")
        return False

if __name__ == "__main__":
    log.info("=== Database Fallback Test ===")
    
    if test_local_postgres():
        log.info("Local PostgreSQL is available for development")
    elif test_sqlite():
        log.info("SQLite is now configured as a fallback database")
    else:
        log.error("Failed to configure any database")
        sys.exit(1)
    
    log.info("=== Database Setup Complete ===")
    sys.exit(0)