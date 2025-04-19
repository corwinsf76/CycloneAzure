# /trading/trader.py

import sys
import os

# Add the project root directory to PYTHONPATH dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import logging
import time
import pandas as pd
import pytz
import datetime
from typing import Dict, Optional, Tuple, List, Any
from binance.client import Client # Import SIDE_BUY, SIDE_SELL etc.
from binance.exceptions import BinanceAPIException, BinanceOrderException, BinanceOrderMinAmountException, BinanceOrderMinPriceException, BinanceOrderMinTotalException, BinanceRequestException # Add missing import for BinanceRequestException

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils
from data_collection import binance_client # Needs function to get current prices
from modeling import predictor # Needs make_prediction function
from trading.portfolio import PortfolioManager # Import the manager class
from feature_engineering.feature_generator import generate_features_for_symbol
from data_collection.technicals import calculate_rsi, calculate_macd
from data_collection.api_data import fetch_market_sentiment, fetch_crypto_health_index, fetch_coin_metrics

from sentiment_analysis.analyzer import get_current_sentiment_score
from sentiment_analysis.advanced_sentiment import get_sentiment_adjusted_prediction
from sentiment_analysis.cross_asset_sentiment import get_cross_asset_adjusted_prediction, get_current_base_sentiments
from trading.contrarian_strategy import get_contrarian_adjusted_prediction
from trading.portfolio import Portfolio
from modeling.predictor import PredictionModel

log = logging.getLogger(__name__)

class Trader:
    """
    Main trader class that combines all trading signals and executes trades.
    """
    
    def __init__(self, portfolio: Portfolio = None):
        """
        Initialize the trader with a portfolio.
        
        Args:
            portfolio: Portfolio object to manage positions and funds
        """
        self.portfolio = portfolio if portfolio else Portfolio()
        self.prediction_model = PredictionModel()
        self.last_trade_decision = {}
        
    async def _get_price_data(self, symbol: str, interval: str, lookback_days: int = 7) -> pd.DataFrame:
        """
        Get recent price data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval (e.g., '1h')
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with price data
        """
        engine = db_utils.engine
        if not engine:
            log.error("Database engine not available")
            return pd.DataFrame()
            
        end_time = datetime.datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=lookback_days)
        
        stmt = f"""
        SELECT open_time as timestamp, open, high, low, close, volume
        FROM price_data
        WHERE symbol = '{symbol}'
          AND interval = '{interval}'
          AND open_time BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY open_time
        """
        
        price_df = pd.read_sql(stmt, engine)
        
        if not price_df.empty:
            # Calculate returns
            price_df['returns'] = price_df['close'].pct_change()
            
            # Set timestamp as index and ensure it's timezone-aware
            price_df.set_index('timestamp', inplace=True)
            price_df.index = pd.to_datetime(price_df.index)
            if price_df.index.tzinfo is None:
                price_df.index = price_df.index.tz_localize(pytz.UTC)
                
        return price_df
    
    async def analyze_trading_opportunity(self, symbol: str) -> Tuple[int, float, Dict[str, Any]]:
        """
        Analyze a potential trading opportunity by combining all signals.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (trade_decision, confidence, metadata)
            where trade_decision is 1 for buy, 0 for sell/hold
        """
        metadata = {}
        
        # Step 1: Get initial ML model prediction
        model_prediction = await self.prediction_model.predict(symbol)
        pred_class, pred_prob = model_prediction
        metadata['initial_prediction'] = {'class': pred_class, 'probability': pred_prob}
        
        log.info(f"Initial prediction for {symbol}: class={pred_class}, probability={pred_prob:.4f}")
        
        # Step 2: Get current sentiment score
        sentiment_score = await get_current_sentiment_score(symbol)
        metadata['sentiment_score'] = sentiment_score
        
        # Step 3: Apply sentiment lag adjustment
        sentiment_adjusted = await get_sentiment_adjusted_prediction(
            symbol, model_prediction, sentiment_score
        )
        metadata['sentiment_adjusted'] = {'class': sentiment_adjusted[0], 'probability': sentiment_adjusted[1]}
        
        # Step 4: Apply cross-asset sentiment adjustment
        # Get current market sentiments for major assets
        current_market_sentiments = await get_current_base_sentiments()
        cross_asset_adjusted = await get_cross_asset_adjusted_prediction(
            symbol, sentiment_adjusted, current_market_sentiments
        )
        metadata['cross_asset_adjusted'] = {'class': cross_asset_adjusted[0], 'probability': cross_asset_adjusted[1]}
        
        # Step 5: Get recent price data for contrarian analysis
        price_data = await self._get_price_data(symbol, config.CANDLE_INTERVAL)
        if price_data.empty:
            log.warning(f"No recent price data available for {symbol}, skipping contrarian analysis")
            final_prediction = cross_asset_adjusted
        else:
            # Apply contrarian strategy adjustment
            contrarian_adjusted = await get_contrarian_adjusted_prediction(
                symbol, cross_asset_adjusted, sentiment_score, price_data
            )
            metadata['contrarian_adjusted'] = {'class': contrarian_adjusted[0], 'probability': contrarian_adjusted[1]}
            final_prediction = contrarian_adjusted
        
        # Get final trade decision and confidence
        final_class, final_prob = final_prediction
        
        # Store the decision for later reference
        self.last_trade_decision[symbol] = {
            'timestamp': datetime.datetime.now(pytz.UTC),
            'decision': final_class,
            'probability': final_prob,
            'metadata': metadata
        }
        
        # Log the full decision process
        log.info(f"Trade analysis complete for {symbol}. Final decision: {final_class} with confidence {final_prob:.4f}")
        
        return final_class, final_prob, metadata
        
    async def execute_trades(self, symbols: List[str], test_mode: bool = True) -> Dict[str, Any]:
        """
        Analyze and execute trades for a list of symbols.
        
        Args:
            symbols: List of trading pair symbols
            test_mode: If True, only simulate trades without actual execution
            
        Returns:
            Dict with trade execution results
        """
        results = {}
        
        for symbol in symbols:
            log.info(f"Analyzing trading opportunity for {symbol}")
            
            decision, confidence, metadata = await self.analyze_trading_opportunity(symbol)
            
            # Only act if confidence is high enough
            min_confidence = config.MIN_TRADE_CONFIDENCE if hasattr(config, 'MIN_TRADE_CONFIDENCE') else 0.65
            
            if confidence >= min_confidence:
                # Execute the trade
                if decision == 1:  # Buy signal
                    amount = config.DEFAULT_POSITION_SIZE if hasattr(config, 'DEFAULT_POSITION_SIZE') else 0.1
                    
                    if test_mode:
                        log.info(f"[TEST MODE] Would BUY {amount} {symbol} with confidence {confidence:.4f}")
                        trade_result = {'status': 'simulated', 'action': 'buy', 'amount': amount}
                    else:
                        trade_result = await self.portfolio.buy(symbol, amount)
                        log.info(f"Executed BUY for {amount} {symbol} with confidence {confidence:.4f}: {trade_result}")
                        
                else:  # Sell signal
                    # Check if we have a position to sell
                    position = self.portfolio.get_position(symbol)
                    
                    if position > 0:
                        if test_mode:
                            log.info(f"[TEST MODE] Would SELL {position} {symbol} with confidence {confidence:.4f}")
                            trade_result = {'status': 'simulated', 'action': 'sell', 'amount': position}
                        else:
                            trade_result = await self.portfolio.sell(symbol, position)
                            log.info(f"Executed SELL for {position} {symbol} with confidence {confidence:.4f}: {trade_result}")
                    else:
                        log.info(f"No position to sell for {symbol}")
                        trade_result = {'status': 'skipped', 'reason': 'no_position'}
            else:
                log.info(f"Confidence too low ({confidence:.4f}) for {symbol}, skipping trade")
                trade_result = {'status': 'skipped', 'reason': 'low_confidence'}
                
            results[symbol] = {
                'decision': decision,
                'confidence': confidence,
                'metadata': metadata,
                'trade_result': trade_result if 'trade_result' in locals() else {'status': 'skipped'}
            }
            
        return results
        
    async def evaluate_strategy_performance(self, symbols: List[str], days: int = 30) -> Dict[str, Any]:
        """
        Evaluate the performance of the trading strategy on historical data.
        
        Args:
            symbols: List of trading pair symbols to evaluate
            days: Number of days to look back for evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        from trading.contrarian_strategy import ContrarianSignalDetector
        
        performance = {}
        detector = ContrarianSignalDetector()
        
        for symbol in symbols:
            log.info(f"Evaluating strategy performance for {symbol}")
            
            # Get historical contrarian signals
            signals_df = await detector.get_historical_contrarian_signals(symbol, lookback_days=days)
            
            if not signals_df.empty:
                # Evaluate contrarian performance
                contrarian_performance = detector.evaluate_contrarian_performance(signals_df)
                
                performance[symbol] = {
                    'contrarian_signals': len(signals_df),
                    'contrarian_performance': contrarian_performance
                }
                
                log.info(f"Contrarian strategy for {symbol} found {len(signals_df)} signals. " +
                         f"Performance: {contrarian_performance}")
            else:
                log.warning(f"No historical signals found for {symbol}")
                performance[symbol] = {'contrarian_signals': 0}
                
        return performance

