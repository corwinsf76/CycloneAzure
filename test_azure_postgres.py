#!/usr/bin/env python3
"""
Azure PostgreSQL Connection Test Script

This script tests the connection to the Azure PostgreSQL database
using the credentials from config.py
"""

import psycopg2
import sys
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_azure_connection():
    """Test connection to Azure PostgreSQL database"""
    logger.info("Testing connection to Azure PostgreSQL database...")
    
    try:
        # Connect to the Azure PostgreSQL database using config
        conn = psycopg2.connect(config.DATABASE_URL)
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Execute a simple query
        cursor.execute("SELECT version();")
        
        # Fetch the result
        version = cursor.fetchone()
        logger.info(f"Successfully connected to Azure PostgreSQL: {version[0]}")
        
        # Check if price_data table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'price_data'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            logger.info("price_data table exists")
            
            # Get a count of records
            cursor.execute("SELECT COUNT(*) FROM price_data;")
            count = cursor.fetchone()[0]
            logger.info(f"price_data table contains {count} records")
        else:
            logger.warning("price_data table does not exist yet")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Azure PostgreSQL: {e}")
        return False

if __name__ == "__main__":
    result = test_azure_connection()
    sys.exit(0 if result else 1)