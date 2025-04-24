#!/usr/bin/env python3
"""
Backfill Script for Low-Value Coin Sentiment Analysis

This script iterates through a historical date range and runs the 
specialized sentiment analysis for low-value coins (< $1) for each day,
storing the aggregated results.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
import pytz

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import config
# Import the historical analysis function (to be created)
from sentiment_analysis.advanced_sentiment import analyze_historical_low_value_sentiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def run_low_value_sentiment_backfill(start_date: datetime, end_date: datetime) -> None:
    """
    Runs the low-value coin sentiment analysis for each day in the specified range.

    Args:
        start_date (datetime): The start date for the backfill (inclusive).
        end_date (datetime): The end date for the backfill (inclusive).
    """
    log.info(f"Starting low-value sentiment backfill from {start_date.date()} to {end_date.date()}")

    current_date = start_date
    while current_date <= end_date:
        log.info(f"Analyzing low-value sentiment for date: {current_date.date()}")
        try:
            # Call the analysis function for the specific day
            await analyze_historical_low_value_sentiment(target_date=current_date)
            log.info(f"Successfully analyzed sentiment for {current_date.date()}")
            
            # Optional: Add a small delay between days if needed
            await asyncio.sleep(1) 
            
        except Exception as e:
            log.error(f"Error analyzing low-value sentiment for {current_date.date()}: {e}", exc_info=True)
            # Decide whether to continue or stop on error
            # For now, log the error and continue with the next day
            await asyncio.sleep(5) # Longer delay on error

        # Move to the next day
        current_date += timedelta(days=1)

    log.info("Finished low-value sentiment backfill.")

async def main():
    """Main function for standalone execution (optional)."""
    # Example: Run for the last 7 days
    end = datetime.now(pytz.UTC)
    start = end - timedelta(days=config.BACKFILL_DAYS) 
    await run_low_value_sentiment_backfill(start_date=start, end_date=end)

if __name__ == "__main__":
    # Example of how to run this script directly
    # In practice, it's called by run_all_backfills.py
    log.info("Running low-value sentiment backfill directly...")
    asyncio.run(main())
    log.info("Direct run finished.")
