"""
Configuration Management System for Cyclone v2 Dashboard

This module provides functionality for:
1. Reading current configuration settings from the config.py
2. Updating configuration settings at runtime
3. Persisting changes to environment variables or database
4. Reloading configuration in running components
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv, set_key
import importlib

# Add the project root directory to PYTHONPATH dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Cyclone v2 imports
import config
from database.db_utils import async_fetch, async_execute
from trading.portfolio import Portfolio

log = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Class for managing Cyclone v2 configuration settings.
    Allows for reading and updating configuration parameters at runtime.
    """
    
    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            env_file_path: Optional path to the .env file. If not provided, 
                           will look for .env in the root directory.
        """
        self.env_file_path = env_file_path or os.path.join(project_root, '.env')
        self.config_module = config
        self.config_cache = {}
        self.portfolio = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load current configuration values into cache"""
        # Load all attributes that don't start with _ and aren't functions
        for attr_name in dir(self.config_module):
            if not attr_name.startswith('_') and not callable(getattr(self.config_module, attr_name)):
                self.config_cache[attr_name] = getattr(self.config_module, attr_name)
    
    def get_config_value(self, key: str) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration parameter name
            
        Returns:
            Configuration value or None if not found
        """
        return self.config_cache.get(key, None)
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return self.config_cache
    
    async def set_config_value(self, key: str, value: Any) -> bool:
        """
        Set a configuration value and persist it.
        
        Args:
            key: Configuration parameter name
            value: New value to set
            
        Returns:
            Boolean indicating success
        """
        try:
            # Convert value to the right type
            current_value = self.get_config_value(key)
            if current_value is not None:
                value_type = type(current_value)
                if value_type in (int, float, bool, str):
                    # Cast to the appropriate type
                    value = value_type(value)
                elif value_type == list:
                    # Handle list values (comma-separated strings)
                    if isinstance(value, str):
                        value = [item.strip() for item in value.split(',')]
            
            # Update in-memory cache
            self.config_cache[key] = value
            
            # Update config module at runtime
            setattr(self.config_module, key, value)
            
            # Persist to environment variable
            env_var_name = key.upper()  # Convention for env vars
            env_value = self._convert_to_env_value(value)
            await self._update_env_file(env_var_name, env_value)
            
            # Save to database for historical tracking
            await self._save_config_change_to_db(key, value)
            
            log.info(f"Updated configuration: {key} = {value}")
            return True
        
        except Exception as e:
            log.error(f"Error updating configuration {key}: {e}")
            return False
    
    def _convert_to_env_value(self, value: Any) -> str:
        """Convert Python value to string for environment variable"""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        else:
            return str(value)
    
    async def _update_env_file(self, key: str, value: str) -> None:
        """Update environment variable in .env file"""
        # This is not async yet, but we'll wrap it to be consistent
        try:
            if os.path.exists(self.env_file_path):
                set_key(self.env_file_path, key, value)
                log.info(f"Updated {key} in {self.env_file_path}")
            else:
                # Create file if it doesn't exist
                with open(self.env_file_path, 'w') as f:
                    f.write(f"{key}={value}\n")
                log.info(f"Created {self.env_file_path} with {key}={value}")
        except Exception as e:
            log.error(f"Error updating .env file: {e}")
    
    async def _save_config_change_to_db(self, key: str, value: Any) -> None:
        """Save configuration change to database for history tracking"""
        try:
            # Create config_changes table if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS config_changes (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                parameter_name VARCHAR(255) NOT NULL,
                parameter_value TEXT NOT NULL,
                change_type VARCHAR(50) NOT NULL
            )
            """
            await async_execute(create_table_sql)
            
            # Insert the change
            insert_sql = """
            INSERT INTO config_changes (parameter_name, parameter_value, change_type)
            VALUES (%s, %s, %s)
            """
            await async_execute(insert_sql, (key, str(value), "dashboard_update"))
            
        except Exception as e:
            log.error(f"Error saving config change to database: {e}")
    
    async def get_config_history(self, parameter_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history of configuration changes.
        
        Args:
            parameter_name: Optional filter by parameter name
            limit: Maximum number of records to return
            
        Returns:
            List of configuration change records
        """
        try:
            if parameter_name:
                query = """
                SELECT * FROM config_changes 
                WHERE parameter_name = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """
                return await async_fetch(query, (parameter_name, limit))
            else:
                query = """
                SELECT * FROM config_changes 
                ORDER BY timestamp DESC
                LIMIT %s
                """
                return await async_fetch(query, (limit,))
        except Exception as e:
            log.error(f"Error retrieving config history: {e}")
            return []
    
    async def get_trading_status(self) -> Dict[str, Any]:
        """
        Get current trading bot status.
        
        Returns:
            Dictionary with trading status information
        """
        try:
            # Check if trading is enabled
            trading_enabled = False
            test_mode = True
            
            # Try to get status from database
            query = """
            SELECT * FROM system_status
            ORDER BY timestamp DESC
            LIMIT 1
            """
            results = await async_fetch(query)
            
            if results:
                status = results[0]
                trading_enabled = status.get('trading_enabled', False)
                test_mode = status.get('test_mode', True)
            
            # Get portfolio summary
            if not self.portfolio:
                self.portfolio = Portfolio()
            
            positions = await self.portfolio.get_all_positions()
            balance = await self.portfolio.get_balance()
            
            return {
                'trading_enabled': trading_enabled,
                'test_mode': test_mode,
                'positions': positions,
                'balance': balance,
                'position_count': len(positions)
            }
            
        except Exception as e:
            log.error(f"Error getting trading status: {e}")
            return {
                'trading_enabled': False,
                'test_mode': True,
                'error': str(e)
            }
    
    async def set_trading_status(self, enabled: bool, test_mode: bool) -> bool:
        """
        Enable or disable trading bot.
        
        Args:
            enabled: Whether trading should be enabled
            test_mode: Whether to use paper trading (test mode) or live trading
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create system_status table if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS system_status (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                trading_enabled BOOLEAN NOT NULL,
                test_mode BOOLEAN NOT NULL,
                updated_by VARCHAR(255) DEFAULT 'dashboard'
            )
            """
            await async_execute(create_table_sql)
            
            # Insert new status
            insert_sql = """
            INSERT INTO system_status (trading_enabled, test_mode)
            VALUES (%s, %s)
            """
            await async_execute(insert_sql, (enabled, test_mode))
            
            # Update config values
            await self.set_config_value('TRADING_ENABLED', enabled)
            await self.set_config_value('TEST_MODE', test_mode)
            
            log.info(f"Trading status updated: enabled={enabled}, test_mode={test_mode}")
            return True
            
        except Exception as e:
            log.error(f"Error setting trading status: {e}")
            return False
    
    async def emergency_shutdown(self) -> Dict[str, Any]:
        """
        Emergency shutdown of the trading bot.
        Sells all positions and disables trading.
        
        Returns:
            Dictionary with shutdown results
        """
        try:
            # Disable trading first
            await self.set_trading_status(enabled=False, test_mode=True)
            
            # Initialize portfolio if not already done
            if not self.portfolio:
                self.portfolio = Portfolio()
            
            # Get current positions
            positions = await self.portfolio.get_all_positions()
            results = {
                'trading_disabled': True,
                'positions_before_shutdown': len(positions),
                'sell_results': []
            }
            
            # Sell all positions
            for symbol, amount in positions.items():
                if amount > 0:
                    try:
                        sell_result = await self.portfolio.sell(symbol, amount)
                        results['sell_results'].append({
                            'symbol': symbol,
                            'amount': amount,
                            'result': sell_result
                        })
                    except Exception as e:
                        results['sell_results'].append({
                            'symbol': symbol,
                            'amount': amount,
                            'error': str(e)
                        })
            
            # Record emergency shutdown in database
            await async_execute(
                "INSERT INTO system_status (trading_enabled, test_mode, updated_by) VALUES (%s, %s, %s)",
                (False, True, 'emergency_shutdown')
            )
            
            # Final check of positions after selling
            remaining_positions = await self.portfolio.get_all_positions()
            results['positions_after_shutdown'] = len(remaining_positions)
            results['remaining_positions'] = remaining_positions
            
            log.warning("EMERGENCY SHUTDOWN EXECUTED - All positions sold and trading disabled")
            return results
            
        except Exception as e:
            log.error(f"Error during emergency shutdown: {e}")
            return {
                'trading_disabled': True,
                'error': str(e),
                'positions_sold': False
            }

# Initialize singleton instance
config_manager = ConfigurationManager()