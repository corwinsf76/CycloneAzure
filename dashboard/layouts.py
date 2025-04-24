"""
Dashboard Layout Module

This module defines the layout components for the dashboard application.
"""

import sys
import os
from typing import List, Dict, Any
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import logging

# Add the project root directory to PYTHONPATH dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from dashboard import data_provider
from dashboard.config_manager import config_manager

log = logging.getLogger(__name__)

def create_layout() -> html.Div:
    """Create the main dashboard layout."""
    
    # Get symbols from config
    symbols = config.BASE_SYMBOLS
    
    # Create tabs for different dashboard sections
    tabs = dbc.Tabs(
        [
            dbc.Tab(label="Trading Dashboard", tab_id="tab-dashboard", children=[
                create_trading_dashboard(symbols)
            ]),
            dbc.Tab(label="Configuration", tab_id="tab-configuration", children=[
                create_configuration_panel()
            ]),
            dbc.Tab(label="Trading Control", tab_id="tab-trading-control", children=[
                create_trading_control_panel()
            ]),
        ],
        id="tabs",
        active_tab="tab-dashboard",
    )
    
    return html.Div([
        # Header
        html.Div([
            html.H1("Cyclone v2 Trading System", 
                   className="text-center mb-2"),
            html.Div([
                html.Span(id="trading-status-badge", className="badge bg-secondary me-2"),
                html.Button(
                    "EMERGENCY STOP", 
                    id="emergency-stop-btn",
                    className="btn btn-danger",
                    title="Sell all positions and disable trading"
                ),
            ], className="d-flex justify-content-end"),
        ], className="container-fluid bg-dark text-white py-3"),

        # Main content with tabs
        html.Div([
            tabs,
            # Store for sharing data between callbacks
            dcc.Store(id='trading-status-store'),
            # Interval for periodic updates
            dcc.Interval(
                id='status-update-interval',
                interval=10000,  # 10 seconds
                n_intervals=0
            ),
            # Confirmation modal for emergency stop
            dbc.Modal(
                [
                    dbc.ModalHeader("Emergency Shutdown Confirmation"),
                    dbc.ModalBody("This will SELL ALL POSITIONS and DISABLE TRADING. Are you sure?"),
                    dbc.ModalFooter([
                        dbc.Button("Cancel", id="emergency-stop-cancel", className="me-2"),
                        dbc.Button("CONFIRM SHUTDOWN", id="emergency-stop-confirm", color="danger"),
                    ]),
                ],
                id="emergency-stop-modal",
                is_open=False,
            ),
            # Success notification
            dbc.Toast(
                "Settings saved successfully!",
                id="settings-toast",
                header="Success",
                is_open=False,
                dismissable=True,
                icon="success",
                duration=4000,
            ),
        ], className="container-fluid p-4"),

        # Footer
        html.Div([
            html.P([
                "Cyclone v2 Trading System • ",
                html.A("Documentation", href="#", className="text-decoration-none")
            ], className="text-center text-muted mb-0")
        ], className="container-fluid bg-light py-3")
    ])

def create_trading_dashboard(symbols: List[str]) -> html.Div:
    """Create the main trading dashboard layout."""
    return html.Div([
        # Controls row
        html.Div([
            html.Div([
                html.Label("Symbol"),
                dcc.Dropdown(
                    id='symbol-selector',
                    options=[{'label': s, 'value': s} for s in symbols],
                    value=symbols[0] if symbols else None,
                    className="mb-3"
                )
            ], className="col-md-3"),
            
            html.Div([
                html.Label("Interval"),
                dcc.Dropdown(
                    id='interval-selector',
                    options=[
                        {'label': '5 minutes', 'value': '5m'},
                        {'label': '15 minutes', 'value': '15m'},
                        {'label': '1 hour', 'value': '1h'},
                        {'label': '4 hours', 'value': '4h'},
                        {'label': '1 day', 'value': '1d'}
                    ],
                    value='5m',
                    className="mb-3"
                )
            ], className="col-md-3"),
            
            html.Div([
                dcc.Interval(
                    id='update-interval',
                    interval=30000,  # 30 seconds
                    n_intervals=0
                )
            ])
        ], className="row mb-4"),

        # Charts row
        html.Div([
            # Price chart
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='price-chart',
                        config={'displayModeBar': True}
                    )
                ], className="card-body")
            ], className="card shadow mb-4 col-md-8"),

            # Sentiment gauges
            html.Div([
                html.Div([
                    html.H5("Sentiment Analysis", className="card-title"),
                    dcc.Graph(
                        id='sentiment-gauge',
                        config={'displayModeBar': False}
                    )
                ], className="card-body mb-3"),
                
                html.Div([
                    dcc.Graph(
                        id='sentiment-timeline',
                        config={'displayModeBar': True}
                    )
                ], className="card-body")
            ], className="card shadow mb-4 col-md-4")
        ], className="row"),

        # Metrics row
        html.Div([
            # Market metrics
            html.Div([
                html.Div([
                    html.H5("Market Metrics", className="card-title"),
                    html.Div(
                        id='market-metrics',
                        className="p-3"
                    )
                ], className="card-body")
            ], className="card shadow mb-4 col-md-4"),

            # Technical indicators
            html.Div([
                html.Div([
                    html.H5("Technical Indicators", className="card-title"),
                    html.Div(
                        id='technical-indicators',
                        className="p-3"
                    )
                ], className="card-body")
            ], className="card shadow mb-4 col-md-4"),

            # Social metrics
            html.Div([
                html.Div([
                    html.H5("Social Metrics", className="card-title"),
                    html.Div(
                        id='social-metrics',
                        className="p-3"
                    )
                ], className="card-body")
            ], className="card shadow mb-4 col-md-4")
        ], className="row"),

        # Low-Value Coin Watchlist Row
        html.Div([
            html.Div([
                 dbc.Card([
                    dbc.CardHeader(html.H5("Low-Value Coin Watchlist (<= $1)", className="mb-0")),
                    dbc.CardBody([
                        dbc.Spinner(html.Div(id="low-value-watchlist-content"))
                    ])
                ])
            ], className="col-12")
        ], className="row mb-4")
    ])

def create_configuration_panel() -> html.Div:
    """Create the configuration settings panel layout."""
    
    # Load current configuration
    config_values = config_manager.get_all_config()
    
    # Extract categories of settings
    trading_params = {
        "TRADE_CAPITAL_PERCENTAGE": "Position Size (%)",
        "STOP_LOSS_PCT": "Stop Loss (%)",
        "TAKE_PROFIT_PCT": "Take Profit (%)",
        "MAX_CONCURRENT_POSITIONS": "Max Concurrent Positions",
        "PORTFOLIO_DRAWDOWN_PCT": "Max Drawdown (%)",
        "BUY_CONFIDENCE_THRESHOLD": "Buy Confidence Threshold",
        "SELL_CONFIDENCE_THRESHOLD": "Sell Confidence Threshold"
    }
    
    low_value_params = {
        "LOW_VALUE_COIN_ENABLED": "Enable Low-Value Coin Strategy",
        "LOW_VALUE_PRICE_THRESHOLD": "Low-Value Price Threshold ($)",
        "LOW_VALUE_COIN_PRIORITY": "Prioritize Low-Value Coins",
        "LOW_VALUE_POSITION_PERCENTAGE": "Low-Value Position Size (%)",
        "LOW_VALUE_SENTIMENT_THRESHOLD": "Low-Value Sentiment Threshold",
        "LOW_VALUE_TWEET_VOLUME_FACTOR": "Low-Value Tweet Volume Factor"
    }
    
    scheduling_params = {
        "PRICE_FETCH_INTERVAL": "Price Fetch Interval (s)",
        "INDICATOR_CALC_INTERVAL": "Indicator Calculation Interval (s)",
        "NEWS_FETCH_INTERVAL": "News Fetch Interval (s)",
        "REDDIT_FETCH_INTERVAL": "Reddit Fetch Interval (s)",
        "TWITTER_FETCH_INTERVAL": "Twitter Fetch Interval (s)",
        "SENTIMENT_ANALYSIS_INTERVAL": "Sentiment Analysis Interval (s)",
        "TRADING_LOGIC_INTERVAL": "Trading Logic Interval (s)",
        "MODEL_RETRAIN_INTERVAL": "Model Retraining Interval (s)"
    }
    
    # Create cards for each category
    return html.Div([
        html.H3("Configuration Settings", className="mb-4"),
        html.P("Adjust trading parameters and system settings. Changes take effect immediately.", className="text-muted mb-4"),
        
        # Trading Parameters Card
        html.Div([
            dbc.Card([
                dbc.CardHeader(html.H5("Risk & Trading Parameters", className="mb-0")),
                dbc.CardBody([
                    create_parameter_inputs(trading_params, config_values, "trading"),
                ])
            ], className="mb-4"),
            
            # Low-Value Coin Strategy Card
            dbc.Card([
                dbc.CardHeader(html.H5("Low-Value Coin Strategy", className="mb-0")),
                dbc.CardBody([
                    create_parameter_inputs(low_value_params, config_values, "low-value"),
                ])
            ], className="mb-4"),
            
            # Scheduling Parameters Card
            dbc.Card([
                dbc.CardHeader(html.H5("Scheduling Parameters", className="mb-0")),
                dbc.CardBody([
                    create_parameter_inputs(scheduling_params, config_values, "scheduling"),
                ])
            ], className="mb-4"),
            
            # Save Button
            html.Div([
                dbc.Button("Save All Settings", id="save-settings-btn", color="primary", className="me-2"),
                dbc.Button("Reset to Defaults", id="reset-settings-btn", color="secondary"),
            ], className="d-flex justify-content-end mt-3")
        ])
    ])

def create_trading_control_panel() -> html.Div:
    """Create the trading control panel layout."""
    return html.Div([
        html.H3("Trading Control Panel", className="mb-4"),
        html.P("Control trading operations and monitor the system status.", className="text-muted mb-4"),
        
        # Trading Status Card
        dbc.Card([
            dbc.CardHeader(html.H5("Trading Status", className="mb-0")),
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.H6("Trading Mode"),
                        dbc.RadioItems(
                            id="trading-mode-select",
                            options=[
                                {"label": "Paper Trading", "value": "paper"},
                                {"label": "Live Trading", "value": "live"},
                            ],
                            value="paper",
                            inline=True,
                            className="mb-3"
                        ),
                    ], className="col-md-6"),
                    
                    html.Div([
                        html.H6("Trading Status"),
                        dbc.Switch(
                            id="trading-enabled-switch",
                            label="Enable Trading",
                            value=False,
                            className="mb-3"
                        ),
                    ], className="col-md-6"),
                ], className="row"),
                
                html.Hr(),
                
                html.Div([
                    html.H6("Trading System Actions", className="mb-3"),
                    dbc.Button("Apply Trading Settings", id="apply-trading-settings-btn", color="primary", className="me-2"),
                    dbc.Button("Sell All Positions", id="sell-all-positions-btn", color="warning", className="me-2"),
                    dbc.Button(
                        [
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            "Emergency Shutdown"
                        ],
                        id="emergency-shutdown-btn",
                        color="danger"
                    ),
                ], className="mt-3"),
            ])
        ], className="mb-4"),
        
        # Portfolio Summary Card
        dbc.Card([
            dbc.CardHeader(html.H5("Portfolio Summary", className="mb-0")),
            dbc.CardBody([
                dbc.Spinner(html.Div(id="portfolio-summary-content")),
                html.Div(id="positions-table-container", className="mt-3"),
            ])
        ], className="mb-4"),
        
        # Recent Trades Card
        dbc.Card([
            dbc.CardHeader(html.H5("Recent Trades", className="mb-0")),
            dbc.CardBody([
                dbc.Spinner(html.Div(id="recent-trades-content")),
            ])
        ]),
    ])

def create_parameter_inputs(params: Dict[str, str], values: Dict[str, Any], prefix: str) -> html.Div:
    """Create input elements for configuration parameters."""
    input_rows = []
    
    for param_key, param_label in params.items():
        current_value = values.get(param_key)
        
        # Skip if param not found
        if current_value is None:
            continue
        
        # Create appropriate input based on value type
        if isinstance(current_value, bool):
            input_element = dbc.Switch(
                id=f"{prefix}-{param_key}",
                value=current_value,
            )
        elif isinstance(current_value, int):
            input_element = dbc.Input(
                id=f"{prefix}-{param_key}",
                type="number",
                value=current_value,
                step=1,
            )
        elif isinstance(current_value, float):
            input_element = dbc.Input(
                id=f"{prefix}-{param_key}",
                type="number",
                value=current_value,
                step=0.01,
            )
        elif isinstance(current_value, list):
            input_element = dbc.Textarea(
                id=f"{prefix}-{param_key}",
                value=", ".join(current_value) if current_value else "",
                rows=2,
            )
        else:
            input_element = dbc.Input(
                id=f"{prefix}-{param_key}",
                type="text",
                value=str(current_value),
            )
        
        # Create a row with label and input
        input_rows.append(
            html.Div([
                html.Div([
                    html.Label(param_label, className="form-label"),
                    dbc.FormText(param_key, color="secondary"),
                ], className="col-md-6"),
                html.Div([
                    input_element
                ], className="col-md-6"),
            ], className="row mb-3")
        )
    
    return html.Div(input_rows)
