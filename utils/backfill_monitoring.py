"""
Monitoring and logging utilities for cryptocurrency data backfill operations.
Provides detailed metrics tracking and structured logging.
"""

import logging
import time
from datetime import datetime
import pytz
from typing import Dict, Any, Optional, List
import json
import os
from functools import wraps
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

class BackfillMetrics:
    """Track and store metrics for backfill operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.items_processed = 0
        self.items_skipped = 0
        self.items_failed = 0
        self.api_calls = 0
        self.api_errors = 0
        self.rate_limit_hits = 0
        self.db_inserts = 0
        self.db_errors = 0
        self.processing_times: List[float] = []
        self.symbols_processed: Dict[str, Dict[str, Any]] = {}
        
    def record_symbol_start(self, symbol: str) -> None:
        """Record the start of processing for a symbol"""
        self.symbols_processed[symbol] = {
            'start_time': time.time(),
            'end_time': None,
            'success': False,
            'items_processed': 0,
            'api_calls': 0,
            'api_errors': 0,
            'db_inserts': 0
        }
        
    def record_symbol_end(self, symbol: str, success: bool) -> None:
        """Record the end of processing for a symbol"""
        if symbol in self.symbols_processed:
            self.symbols_processed[symbol]['end_time'] = time.time()
            self.symbols_processed[symbol]['success'] = success
            
    def record_api_call(self, symbol: Optional[str] = None) -> None:
        """Record an API call"""
        self.api_calls += 1
        if symbol and symbol in self.symbols_processed:
            self.symbols_processed[symbol]['api_calls'] += 1
            
    def record_api_error(self, symbol: Optional[str] = None) -> None:
        """Record an API error"""
        self.api_errors += 1
        if symbol and symbol in self.symbols_processed:
            self.symbols_processed[symbol]['api_errors'] += 1
            
    def record_rate_limit_hit(self) -> None:
        """Record a rate limit hit"""
        self.rate_limit_hits += 1
        
    def record_db_insert(self, count: int = 1, symbol: Optional[str] = None) -> None:
        """Record database insertions"""
        self.db_inserts += count
        if symbol and symbol in self.symbols_processed:
            self.symbols_processed[symbol]['db_inserts'] += count
            
    def record_db_error(self) -> None:
        """Record a database error"""
        self.db_errors += 1
        
    def record_processing_time(self, seconds: float) -> None:
        """Record processing time for an operation"""
        self.processing_times.append(seconds)
        
    def record_item_processed(self, count: int = 1, symbol: Optional[str] = None) -> None:
        """Record processed items"""
        self.items_processed += count
        if symbol and symbol in self.symbols_processed:
            self.symbols_processed[symbol]['items_processed'] += count
            
    def record_item_skipped(self, count: int = 1) -> None:
        """Record skipped items"""
        self.items_skipped += count
        
    def record_item_failed(self, count: int = 1) -> None:
        """Record failed items"""
        self.items_failed += count
        
    def complete(self) -> None:
        """Mark the operation as complete"""
        self.end_time = time.time()
        
    def get_duration(self) -> float:
        """Get the total duration of the operation in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
        
    def get_avg_processing_time(self) -> Optional[float]:
        """Get the average processing time"""
        if not self.processing_times:
            return None
        return sum(self.processing_times) / len(self.processing_times)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metrics"""
        return {
            'operation_name': self.operation_name,
            'start_time': datetime.fromtimestamp(self.start_time, pytz.UTC).isoformat(),
            'end_time': datetime.fromtimestamp(self.end_time, pytz.UTC).isoformat() if self.end_time else None,
            'duration_seconds': self.get_duration(),
            'items_processed': self.items_processed,
            'items_skipped': self.items_skipped,
            'items_failed': self.items_failed,
            'success_rate': (self.items_processed / (self.items_processed + self.items_failed)) * 100 if (self.items_processed + self.items_failed) > 0 else 0,
            'api_calls': self.api_calls,
            'api_errors': self.api_errors,
            'api_error_rate': (self.api_errors / self.api_calls) * 100 if self.api_calls > 0 else 0,
            'rate_limit_hits': self.rate_limit_hits,
            'db_inserts': self.db_inserts,
            'db_errors': self.db_errors,
            'avg_processing_time': self.get_avg_processing_time(),
            'symbols_count': len(self.symbols_processed)
        }
        
    def log_summary(self) -> None:
        """Log a summary of the metrics"""
        summary = self.get_summary()
        logger.info(f"--- {self.operation_name} Summary ---")
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info(f"Items: {summary['items_processed']} processed, {summary['items_skipped']} skipped, {summary['items_failed']} failed")
        logger.info(f"Success rate: {summary['success_rate']:.2f}%")
        logger.info(f"API calls: {summary['api_calls']} total, {summary['api_errors']} errors ({summary['api_error_rate']:.2f}%)")
        logger.info(f"Rate limit hits: {summary['rate_limit_hits']}")
        logger.info(f"Database: {summary['db_inserts']} inserts, {summary['db_errors']} errors")
        logger.info(f"Symbols processed: {summary['symbols_count']}")
        logger.info("-----------------------")
        
    def save_to_file(self, directory: str = 'logs/metrics') -> None:
        """Save metrics to a JSON file"""
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M%S')
        filename = f"{directory}/{self.operation_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        logger.info(f"Metrics saved to {filename}")

# Decorator for tracking API function metrics
def track_api_metrics(metrics: BackfillMetrics, symbol_param_name: str = 'symbol'):
    """
    Decorator to track metrics for API functions
    
    Args:
        metrics: The BackfillMetrics instance to update
        symbol_param_name: The name of the parameter containing the symbol
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the symbol from the arguments
            symbol = kwargs.get(symbol_param_name)
            if not symbol and args and len(args) > 0:
                # Try to get symbol from positional args based on function signature
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if symbol_param_name in params:
                    idx = params.index(symbol_param_name)
                    if idx < len(args):
                        symbol = args[idx]
            
            start_time = time.time()
            metrics.record_api_call(symbol)
            
            try:
                result = await func(*args, **kwargs)
                processing_time = time.time() - start_time
                metrics.record_processing_time(processing_time)
                
                if result:
                    # Assume success
                    if hasattr(result, '__len__'):
                        metrics.record_item_processed(len(result), symbol)
                    else:
                        metrics.record_item_processed(1, symbol)
                return result
            except Exception as e:
                metrics.record_api_error(symbol)
                if "rate limit" in str(e).lower():
                    metrics.record_rate_limit_hit()
                raise
                
        return wrapper
    return decorator

# Decorator for tracking database operations
def track_db_metrics(metrics: BackfillMetrics):
    """Decorator to track metrics for database operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                
                # Try to determine the number of records inserted
                records = None
                for arg in args:
                    if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], dict):
                        records = arg
                        break
                    
                if records:
                    metrics.record_db_insert(len(records))
                else:
                    metrics.record_db_insert(1)
                    
                return result
            except Exception as e:
                metrics.record_db_error()
                raise
                
        return wrapper
    return decorator

async def log_backfill_progress(metrics: BackfillMetrics, interval_seconds: int = 60):
    """
    Periodically log progress during backfill operations
    
    Args:
        metrics: The BackfillMetrics instance to monitor
        interval_seconds: How often to log progress
    """
    while True:
        if metrics.end_time:
            # Operation is complete
            break
            
        summary = metrics.get_summary()
        logger.info(f"Progress - {metrics.operation_name}: {summary['items_processed']} items processed, "
                   f"{summary['api_calls']} API calls, {summary['db_inserts']} DB inserts, "
                   f"running for {summary['duration_seconds']:.2f}s")
        
        await asyncio.sleep(interval_seconds)