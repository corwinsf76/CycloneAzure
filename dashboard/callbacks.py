"""
Dashboard callbacks module with async support

This module defines the callback functions for the dashboard application,
including configuration management, trading control, and data visualization.
"""

from typing import List, Dict, Any
import logging
import json
import asyncio
import pandas as pd
from dash import Input, Output, State, ctx, callback, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dash_table

from .data_provider import (
    get_price_data,
    get_sentiment_data,
    get_market_metrics,
    get_technical_indicators,
    get_social_metrics,
    get_sentiment_history,
    get_low_value_coins,
    get_portfolio_holdings
)
from .config_manager import config_manager
from trading.portfolio import Portfolio

log = logging.getLogger(__name__)

def register_callbacks(app):
    @app.callback(
        Output('price-chart', 'figure'),
        Output('sentiment-gauge', 'figure'),
        Output('sentiment-timeline', 'figure'),
        Output('market-metrics', 'children'),
        Output('technical-indicators', 'children'),
        Output('social-metrics', 'children'),
        Input('update-interval', 'n_intervals'),
        State('symbol-selector', 'value')
    )
    def update_dashboard(n_intervals: int, symbol: str):
        """
        Main dashboard update callback - converted from async to sync
        Uses asyncio.run() to execute async data fetching
        """
        if not symbol:
            raise PreventUpdate

        try:
            # Define async function to fetch all data concurrently
            async def fetch_all_data():
                price_df_task = get_price_data(symbol)
                sentiment_data_task = get_sentiment_data(symbol)
                sentiment_history_task = get_sentiment_history(symbol, hours=72)
                market_data_task = get_market_metrics(symbol)
                technical_data_task = get_technical_indicators(symbol)
                social_data_task = get_social_metrics(symbol)

                return await asyncio.gather(
                    price_df_task,
                    sentiment_data_task,
                    sentiment_history_task,
                    market_data_task,
                    technical_data_task,
                    social_data_task
                )

            # Execute async function synchronously
            price_df, sentiment_data, sentiment_history_df, market_data, technical_data, social_data = asyncio.run(fetch_all_data())

            # Create price chart
            price_fig = go.Figure(data=[
                go.Candlestick(
                    x=price_df['open_time'],
                    open=price_df['open'],
                    high=price_df['high'],
                    low=price_df['low'],
                    close=price_df['close']
                )
            ])
            price_fig.update_layout(
                title=f'{symbol} Price Action',
                yaxis_title='Price',
                template='plotly_dark'
            )

            # Create sentiment gauge
            sentiment_fig = create_sentiment_gauge(sentiment_data)

            # Create sentiment timeline chart
            sentiment_timeline_fig = create_sentiment_timeline_chart(sentiment_history_df, symbol)

            # Create market metrics card
            market_card = create_market_metrics_card(market_data)

            # Create technical indicators card
            technical_card = create_technical_card(technical_data)

            # Create social metrics card
            social_card = create_social_metrics_card(social_data)

            return price_fig, sentiment_fig, sentiment_timeline_fig, market_card, technical_card, social_card

        except Exception as e:
            log.error(f"Error updating dashboard for {symbol}: {e}")
            return go.Figure(), go.Figure(), go.Figure(), [], [], []

    @app.callback(
        Output('social-feed', 'children'),
        Input('status-update-interval', 'n_intervals'),
        State('symbol-selector', 'value')
    )
    def update_social_feed(n_intervals: int, symbol: str):
        """
        Update social media feed - converted from async to sync
        """
        if not symbol:
            raise PreventUpdate
            
        try:
            # Execute async function synchronously
            sentiment_data = asyncio.run(get_sentiment_data(symbol))
            
            if not sentiment_data or not sentiment_data.get('raw_data'):
                return "No social data available"
                
            feed_items = []
            
            # Process news items
            for item in sentiment_data['raw_data'].get('news', []):
                feed_items.append(
                    create_news_card(item)
                )
            
            # Process social items
            for item in sentiment_data['raw_data'].get('social', []):
                feed_items.append(
                    create_social_card(item)
                )
            
            return feed_items
            
        except Exception as e:
            log.error(f"Error updating social feed: {e}")
            raise PreventUpdate

    @app.callback(
        Output('trading-status-store', 'data'),
        Output('trading-status-badge', 'children'),
        Output('trading-status-badge', 'className'),
        Output('trading-enabled-switch', 'value'),
        Output('trading-mode-select', 'value'),
        Input('status-update-interval', 'n_intervals')
    )
    def update_trading_status(n_intervals):
        """
        Periodically fetch and update the trading status - converted from async to sync
        """
        try:
            # Execute async function synchronously
            status = asyncio.run(config_manager.get_trading_status())
            
            trading_enabled = status.get('trading_enabled', False)
            test_mode = status.get('test_mode', True)
            
            if trading_enabled:
                if test_mode:
                    badge_text = "PAPER TRADING ACTIVE"
                    badge_class = "badge bg-warning text-dark me-2"
                else:
                    badge_text = "LIVE TRADING ACTIVE"
                    badge_class = "badge bg-success me-2"
            else:
                badge_text = "TRADING DISABLED"
                badge_class = "badge bg-secondary me-2"
            
            switch_value = trading_enabled
            mode_value = "paper" if test_mode else "live"
            
            return status, badge_text, badge_class, switch_value, mode_value
            
        except Exception as e:
            log.error(f"Error updating trading status: {e}")
            return {}, "STATUS ERROR", "badge bg-danger me-2", False, "paper"

    @app.callback(
        Output('low-value-watchlist-content', 'children'),
        Input('update-interval', 'n_intervals')
    )
    def update_low_value_watchlist(n_intervals):
        """
        Fetch and display the low-value coin watchlist. - converted to sync
        """
        try:
            # Execute async function synchronously
            low_value_coins = asyncio.run(get_low_value_coins())
            if not low_value_coins:
                return html.P("No low-value coins found matching the criteria.", className="text-muted")

            return create_low_value_table(low_value_coins)

        except Exception as e:
            log.error(f"Error updating low-value watchlist: {e}")
            return html.P("Error loading low-value coin data.", className="text-danger")

    @app.callback(
        Output('portfolio-summary-content', 'children'),
        Output('positions-table-container', 'children'),
        Input('trading-status-store', 'data'),
        Input('tabs', 'active_tab')
    )
    async def update_portfolio_summary(status_data, active_tab):
        """
        Update portfolio summary and holdings table.
        Fetches holdings directly from the database.
        """
        if active_tab != "tab-trading-control":
            return no_update, no_update

        try:
            holdings_df = await get_portfolio_holdings()

            balance = status_data.get('balance', 0) if status_data else 0
            position_count = len(holdings_df)
            trading_enabled = status_data.get('trading_enabled', False) if status_data else False

            summary_content = [
                html.Div([
                    html.Div([
                        html.H6("Portfolio Balance"),
                        html.H3(f"${balance:,.2f}", className="text-primary")
                    ], className="col-md-4"),

                    html.Div([
                        html.H6("Open Positions"),
                        html.H3(f"{position_count}", className="text-info")
                    ], className="col-md-4"),

                    html.Div([
                        html.H6("Trading Status"),
                        html.H3([
                            html.Span(
                                "ACTIVE" if trading_enabled else "INACTIVE",
                                className=f"badge {'bg-success' if trading_enabled else 'bg-secondary'}"
                            )
                        ])
                    ], className="col-md-4"),
                ], className="row text-center")
            ]

            if holdings_df.empty:
                positions_table = html.P("No active positions", className="text-muted")
            else:
                positions_table = dash_table.DataTable(
                    id='positions-table',
                    columns=[
                        {'name': col, 'id': col} for col in holdings_df.columns
                    ],
                    data=holdings_df.to_dict('records'),
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'color': 'white'
                    },
                    style_cell={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white',
                        'textAlign': 'left'
                    },
                    style_as_list_view=True,
                    page_size=10,
                )

            return summary_content, positions_table

        except Exception as e:
            log.error(f"Error updating portfolio summary: {e}")
            error_message = html.Div([
                html.P(f"Error loading portfolio data: {str(e)}", className="text-danger")
            ])
            return error_message, html.Div()

    @app.callback(
        Output('settings-toast', 'is_open'),
        Input('save-settings-btn', 'n_clicks'),
        State({'type': 'config-input', 'id': ALL}, 'id'),
        State({'type': 'config-input', 'id': ALL}, 'value'),
        prevent_initial_call=True
    )
    async def save_configuration_settings(n_clicks, input_ids, input_values):
        """
        Save all modified configuration parameters
        """
        if not n_clicks:
            return False
            
        try:
            for input_id, value in zip(input_ids, input_values):
                param_key = input_id['id']
                await config_manager.set_config_value(param_key, value)
                
            log.info(f"Saved {len(input_ids)} configuration parameters")
            return True
            
        except Exception as e:
            log.error(f"Error saving configuration: {e}")
            return False

    @app.callback(
        Output('trading-status-store', 'data', allow_duplicate=True),
        Input('apply-trading-settings-btn', 'n_clicks'),
        State('trading-enabled-switch', 'value'),
        State('trading-mode-select', 'value'),
        prevent_initial_call=True
    )
    async def apply_trading_settings(n_clicks, trading_enabled, trading_mode):
        """
        Apply trading status settings (enabled/disabled and paper/live mode)
        """
        if not n_clicks:
            raise PreventUpdate
            
        try:
            test_mode = trading_mode == "paper"
            
            success = await config_manager.set_trading_status(
                enabled=trading_enabled,
                test_mode=test_mode
            )
            
            if success:
                new_status = await config_manager.get_trading_status()
                return new_status
            else:
                raise Exception("Failed to update trading status")
                
        except Exception as e:
            log.error(f"Error applying trading settings: {e}")
            raise PreventUpdate

    @app.callback(
        Output('trading-status-store', 'data', allow_duplicate=True),
        Input('sell-all-positions-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    async def sell_all_positions(n_clicks):
        """
        Sell all currently held positions
        """
        if not n_clicks:
            raise PreventUpdate
            
        try:
            portfolio = config_manager.portfolio
            if not portfolio:
                portfolio = config_manager.portfolio = Portfolio()
                
            positions = await portfolio.get_all_positions()
            
            for symbol, amount in positions.items():
                if amount > 0:
                    await portfolio.sell(symbol, amount)
                    log.info(f"Sold {amount} of {symbol}")
            
            new_status = await config_manager.get_trading_status()
            return new_status
            
        except Exception as e:
            log.error(f"Error selling all positions: {e}")
            raise PreventUpdate

    @app.callback(
        Output('emergency-stop-modal', 'is_open'),
        Input('emergency-stop-btn', 'n_clicks'),
        Input('emergency-shutdown-btn', 'n_clicks'),
        Input('emergency-stop-cancel', 'n_clicks'),
        Input('emergency-stop-confirm', 'n_clicks'),
        State('emergency-stop-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_emergency_modal(btn1, btn2, cancel, confirm, is_open):
        """
        Toggle the emergency shutdown confirmation modal
        """
        if btn1 or btn2:
            return True
        elif cancel:
            return False
        elif confirm:
            return False
        return is_open

    @app.callback(
        Output('trading-status-store', 'data', allow_duplicate=True),
        Input('emergency-stop-confirm', 'n_clicks'),
        prevent_initial_call=True
    )
    async def execute_emergency_shutdown(n_clicks):
        """
        Execute emergency shutdown when confirmed
        """
        if not n_clicks:
            raise PreventUpdate
            
        try:
            shutdown_result = await config_manager.emergency_shutdown()
            
            log.warning(f"Emergency shutdown executed: {shutdown_result}")
            
            status = await config_manager.get_trading_status()
            return status
            
        except Exception as e:
            log.error(f"Error during emergency shutdown: {e}")
            raise PreventUpdate

    @app.callback(
        Output({'type': 'config-input', 'id': ALL}, 'value'),
        Input('reset-settings-btn', 'n_clicks'),
        State({'type': 'config-input', 'id': ALL}, 'id'),
        prevent_initial_call=True
    )
    def reset_to_defaults(n_clicks, input_ids):
        """
        Reset configuration parameters to default values
        """
        if not n_clicks:
            raise PreventUpdate
            
        try:
            default_values = {}
            
            current_values = []
            for param_id in input_ids:
                param_key = param_id['id']
                current_values.append(config_manager.get_config_value(param_key))
                
            return current_values
            
        except Exception as e:
            log.error(f"Error resetting to defaults: {e}")
            raise PreventUpdate

def create_sentiment_gauge(sentiment_data: Dict[str, Any]) -> go.Figure:
    """Create a sentiment gauge chart"""
    current_sentiment = 0.0
    if sentiment_data and sentiment_data.get('aggregates'):
        if 'social' in sentiment_data['aggregates'] and sentiment_data['aggregates']['social']['current'] != 0.0:
             current_sentiment = sentiment_data['aggregates']['social']['current']
        elif 'news' in sentiment_data['aggregates'] and sentiment_data['aggregates']['news']['current'] != 0.0:
             current_sentiment = sentiment_data['aggregates']['news']['current']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_sentiment,
        title={'text': "Overall Sentiment"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0], 'color': "orange"},
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1], 'color': "green"}
            ]
        }
    ))

    fig.update_layout(template='plotly_dark', height=250, margin=dict(t=40, b=40, l=20, r=20))
    return fig

def create_sentiment_timeline_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a time-series chart for sentiment scores."""
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['score'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='cyan'),
            marker=dict(size=4)
        ))
        if len(df) > 5:
             df['rolling_avg'] = df['score'].rolling(window=5, center=True).mean()
             fig.add_trace(go.Scatter(
                 x=df['timestamp'],
                 y=df['rolling_avg'],
                 mode='lines',
                 name='Rolling Avg (5pt)',
                 line=dict(color='magenta', dash='dash')
             ))

    fig.update_layout(
        title=f"{symbol} Sentiment Over Time",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        yaxis_range=[-1, 1],
        template='plotly_dark',
        height=300,
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_low_value_table(coins_data: List[Dict[str, Any]]) -> dash_table.DataTable:
    """Create a DataTable for the low-value coin watchlist."""
    df = pd.DataFrame(coins_data)
    df = df.rename(columns={'symbol': 'Symbol', 'current_price': 'Price ($)', 'name': 'Name'})

    return dash_table.DataTable(
        id='low-value-table',
        columns=[{'name': i, 'id': i} for i in df.columns],
        data=df.to_dict('records'),
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_cell={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white',
            'textAlign': 'left',
            'padding': '5px'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Price ($)'},
                'textAlign': 'right'
            }
        ],
        style_as_list_view=True,
        page_size=10,
        sort_action='native',
        filter_action='native',
    )

def create_market_metrics_card(market_data: Dict[str, Any]) -> List:
    """Create market metrics display"""
    if not market_data:
        return ["No market data available"]
        
    return [
        f"Market Cap: ${market_data['market_cap']:,.2f}",
        f"24h Volume: ${market_data['volume_24h']:,.2f}",
        f"24h Change: {market_data['price_change_24h']:.2f}%",
        f"Market Rank: #{market_data['market_rank']}",
        f"Community Score: {market_data['community_score']:.1f}"
    ]

def create_technical_card(technical_data: Dict[str, Any]) -> List:
    """Create technical indicators display"""
    if not technical_data:
        return ["No technical data available"]
        
    return [
        f"RSI (14): {technical_data['rsi']:.2f}",
        f"MACD: {technical_data['macd']['value']:.2f}",
        f"Signal: {technical_data['macd']['signal']:.2f}",
        f"SMA 20: {technical_data['moving_averages']['sma_20']:.2f}",
        f"SMA 50: {technical_data['moving_averages']['sma_50']:.2f}",
        f"SMA 200: {technical_data['moving_averages']['sma_200']:.2f}"
    ]

def create_social_metrics_card(social_data: Dict[str, Any]) -> List:
    """Create social metrics display"""
    if not social_data:
        return ["No social data available"]
        
    twitter_data = social_data.get('twitter', {})
    reddit_data = social_data.get('reddit', {})
    
    return [
        "Twitter Metrics:",
        f"Posts: {twitter_data.get('post_count', 0)}",
        f"Engagement: {twitter_data.get('total_engagement', 0)}",
        f"Sentiment: {twitter_data.get('avg_sentiment', 0):.2f}",
        "Reddit Metrics:",
        f"Posts: {reddit_data.get('post_count', 0)}",
        f"Comments: {reddit_data.get('total_comments', 0)}",
        f"Score: {reddit_data.get('total_score', 0)}",
        f"Sentiment: {reddit_data.get('avg_sentiment', 0):.2f}"
    ]

def create_news_card(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create a news item card"""
    return {
        'type': 'news',
        'timestamp': item['timestamp'],
        'score': item['score'],
        'source': item['source']
    }

def create_social_card(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create a social media item card"""
    return {
        'type': 'social',
        'timestamp': item['timestamp'],
        'score': item['score'],
        'platform': item['platform']
    }
