"""
Main dashboard application module with async support
"""

import asyncio
from typing import Dict, Any
import logging
import os
import sys
from dash import Dash, html
import dash_bootstrap_components as dbc
from flask import Flask

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dashboard.layouts import create_layout
from dashboard.callbacks import register_callbacks
from dashboard.config_manager import config_manager
from dashboard.data_provider import (
    get_price_data,
    get_sentiment_data,
    get_market_metrics,
    get_technical_indicators,
    get_social_metrics
)

log = logging.getLogger(__name__)

# Initialize Flask for server
server = Flask(__name__)

# Initialize the resources needed by the dashboard
def init_resources():
    """
    Initialize resources for the dashboard
    """
    log.info("Initializing dashboard resources...")
    
    # Initialize config_manager and ensure DB tables exist
    try:
        # Create necessary database tables for config tracking
        # This will be handled by the config_manager when it's first used
        log.info("Configuration management system initialized")
    except Exception as e:
        log.error(f"Error initializing configuration system: {e}")

# Initialize resources immediately instead of waiting for first request
init_resources()

# Initialize Dash with Bootstrap and Font Awesome for icons
app = Dash(
    __name__,
    server=server,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.FONT_AWESOME  # Add Font Awesome for icons
    ],
    suppress_callback_exceptions=True
)

# Set page title
app.title = "Cyclone v2 Trading System"

# Create the layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

def update_data_cache(symbol: str) -> Dict[str, Any]:
    """
    Update all data sources
    """
    try:
        # We'll handle async operations differently with Flask
        # For now, just return an empty dict as a placeholder
        return {}
        
    except Exception as e:
        log.error(f"Error updating data cache: {e}")
        return {}

# Setup proper entry point for running the dashboard
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the Flask server with Dash mounted on it
    server.run(host='0.0.0.0', port=8053, debug=True)
