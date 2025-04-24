#!/usr/bin/env python
"""
Dashboard Launcher with improved error handling and setup

This script launches the Cyclone v2 dashboard with proper logging
and error handling to help diagnose loading issues.
"""

import os
import sys
import logging
import asyncio
import traceback
from pathlib import Path

# Ensure we can import from the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check for required environment
try:
    import dash
    import dash_bootstrap_components as dbc
    import quart
    import pandas as pd
    import plotly
    import asyncpg
except ImportError as e:
    print(f"CRITICAL ERROR: Required package not found: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, 'dashboard.log'), mode='w')
    ]
)
log = logging.getLogger("dashboard_launcher")

# Import configuration to verify it loads correctly
try:
    import config
    log.info(f"Configuration loaded. Log level: {config.LOG_LEVEL}")
    
    # Set the database URL from config
    log.info(f"Database URL configured: {config.DATABASE_URL[:20]}...")
    
    # Change global log level if specified
    if hasattr(config, 'LOG_LEVEL'):
        logging.getLogger().setLevel(config.LOG_LEVEL)
        log.info(f"Log level set to: {config.LOG_LEVEL}")
except ImportError as e:
    log.critical(f"Failed to import config: {e}")
    sys.exit(1)
except Exception as e:
    log.critical(f"Error loading configuration: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test database connectivity early to identify connection issues
async def test_db_connection():
    """Simple test for database connectivity"""
    try:
        from database.db_utils import get_pool, async_fetch
        
        log.info("Testing database connection...")
        pool = await get_pool()
        if not pool:
            log.error("Failed to create database connection pool")
            return False
        
        result = await async_fetch("SELECT 1 as test")
        if result and result[0]['test'] == 1:
            log.info("Database connection successful!")
            return True
        else:
            log.error("Database query failed")
            return False
    except Exception as e:
        log.error(f"Database connection error: {e}")
        traceback.print_exc()
        return False

from dashboard.low_value_analyzer import LowValueAnalyzer

class Dashboard:
    def __init__(self):
        self.low_value_analyzer = LowValueAnalyzer(self.pool)

    async def get_dashboard_data(self):
        low_value_coins = await self.low_value_analyzer.get_low_value_coins()
        portfolio = await self.low_value_analyzer.get_portfolio_holdings()
        
        # Get sentiment for top 5 low value coins
        sentiment_data = {}
        for coin in low_value_coins[:5]:
            sentiment_data[coin['coin_id']] = await self.low_value_analyzer.get_sentiment_history(coin['coin_id'])
        
        return {
            'low_value_coins': low_value_coins,
            'sentiment_data': sentiment_data,
            'portfolio': portfolio
        }

def main():
    """Main entry point for the dashboard launcher"""
    log.info("Starting Cyclone v2 Dashboard")
    
    # Run database test
    try:
        db_ok = asyncio.run(test_db_connection())
        if not db_ok:
            log.warning("Database connection test failed - dashboard may not function correctly")
    except Exception as e:
        log.error(f"Error testing database: {e}")
        traceback.print_exc()
    
    # Print environment info
    log.info(f"Python version: {sys.version}")
    log.info(f"Dash version: {dash.__version__}")
    log.info(f"Project root: {project_root}")
    
    # Start the dashboard
    try:
        log.info("Importing dashboard app...")
        from dashboard.app import server, app
        
        log.info("Dashboard app initialized, starting server...")
        server.run(host='0.0.0.0', port=8052, debug=True)
    except ImportError as e:
        log.critical(f"Failed to import dashboard app: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        log.critical(f"Error launching dashboard: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()