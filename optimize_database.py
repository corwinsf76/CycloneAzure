"""
Database optimization task for cryptocurrency data.
This script runs partitioning and aggregation tasks on the database.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import pytz
from typing import List, Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

import config
from database.db_utils import get_db_pool
from database.timeseries_storage import TimeSeriesPartitioner, TimeSeriesAggregator
from utils.backfill_monitoring import BackfillMetrics, log_backfill_progress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def setup_database_optimizations(force_rebuild: bool = False) -> bool:
    """
    Set up database partitioning and aggregation for efficient time series storage.
    
    Args:
        force_rebuild: Whether to force rebuilding tables (dangerous!)
        
    Returns:
        Success status
    """
    # Create metrics tracker
    metrics = BackfillMetrics("database_optimization")
    
    try:
        # Start progress logging in background
        progress_task = asyncio.create_task(log_backfill_progress(metrics, interval_seconds=30))
        
        # Get database connection pool
        pool = await get_db_pool()
        
        # Set up partitioning for price_data
        price_partitioner = TimeSeriesPartitioner(pool, 'price_data', 'monthly')
        price_partition_success = await price_partitioner.setup_partitioning()
        
        if not price_partition_success:
            log.error("Failed to set up partitioning for price_data")
            metrics.record_item_failed(1)
        else:
            metrics.record_item_processed(1)
            log.info("Successfully set up partitioning for price_data")
            
        # Create partitions for recent months
        now = datetime.now(pytz.UTC)
        current_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Create partitions for last 3 months and next month
        for i in range(-2, 2):
            target_month = current_month.replace(month=((current_month.month + i - 1) % 12) + 1)
            if i > 0 and target_month.month < current_month.month:
                target_month = target_month.replace(year=current_month.year + 1)
            elif i < 0 and target_month.month > current_month.month:
                target_month = target_month.replace(year=current_month.year - 1)
                
            partition_name = await price_partitioner.create_partition_for_date(target_month)
            if partition_name:
                metrics.record_item_processed(1)
                log.info(f"Created partition {partition_name}")
            else:
                metrics.record_item_failed(1)
                
        # Set up partitioning for api_data
        api_partitioner = TimeSeriesPartitioner(pool, 'api_data', 'monthly')
        api_partition_success = await api_partitioner.setup_partitioning()
        
        if not api_partition_success:
            log.error("Failed to set up partitioning for api_data")
            metrics.record_item_failed(1)
        else:
            metrics.record_item_processed(1)
            log.info("Successfully set up partitioning for api_data")
            
        # Create API data partitions for recent months
        for i in range(-2, 2):
            target_month = current_month.replace(month=((current_month.month + i - 1) % 12) + 1)
            if i > 0 and target_month.month < current_month.month:
                target_month = target_month.replace(year=current_month.year + 1)
            elif i < 0 and target_month.month > current_month.month:
                target_month = target_month.replace(year=current_month.year - 1)
                
            partition_name = await api_partitioner.create_partition_for_date(target_month)
            if partition_name:
                metrics.record_item_processed(1)
                log.info(f"Created partition {partition_name}")
            else:
                metrics.record_item_failed(1)
                
        # Set up aggregation
        aggregator = TimeSeriesAggregator(pool)
        aggregation_setup_success = await aggregator.setup_aggregation_tables()
        
        if not aggregation_setup_success:
            log.error("Failed to set up aggregation tables")
            metrics.record_item_failed(1)
        else:
            metrics.record_item_processed(1)
            log.info("Successfully set up aggregation tables")
            
        # Run initial aggregation
        aggregation_results = await aggregator.aggregate_data(force_full=force_rebuild)
        metrics.record_item_processed(sum(aggregation_results.values()))
        
        # Summarize results
        log.info(f"Aggregated {aggregation_results['hourly']} hourly, {aggregation_results['daily']} daily records")
        
        # Complete metrics tracking
        metrics.complete()
        metrics.log_summary()
        metrics.save_to_file()
        
        # Cancel progress logging
        progress_task.cancel()
        
        return True
        
    except Exception as e:
        log.error(f"Error during database optimization: {e}", exc_info=True)
        metrics.complete()
        metrics.log_summary()
        metrics.save_to_file()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up database optimizations for cryptocurrency data")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of aggregation tables (warning: data loss)")
    args = parser.parse_args()
    
    try:
        asyncio.run(setup_database_optimizations(force_rebuild=args.force_rebuild))
    except KeyboardInterrupt:
        log.info("Database optimization interrupted by user")
    except Exception as e:
        log.critical(f"Unhandled error during database optimization: {e}", exc_info=True)
        sys.exit(1)