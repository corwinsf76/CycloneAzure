#!/usr/bin/env python3
"""
Analyzes sentiment for backfilled news and social media data.

This script queries data stored by backfill scripts (news_data, social_media_data)
that hasn't had sentiment analysis applied yet, runs the analysis,
and updates the records in the database.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
import pytz
from typing import List, Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import config
from database.db_utils import async_fetch, async_execute
from sentiment_analysis.analyzer import analyze_texts_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

async def check_and_create_table(table_name: str, text_column: str, id_column: str) -> bool:
    """
    Checks if a table exists and creates it if it doesn't.
    Returns True if table exists or was created successfully, False otherwise.
    """
    # Check if table exists
    table_exists_query = f"""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = '{table_name}'
    )
    """
    
    try:
        result = await async_fetch(table_exists_query)
        if not result or not result[0]['exists']:
            log.info(f"Table {table_name} does not exist. Creating it...")
            
            # Create table with necessary columns
            create_table_query = f"""
            CREATE TABLE {table_name} (
                {id_column} SERIAL PRIMARY KEY,
                {text_column} TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                sentiment_score NUMERIC(10, 6),
                sentiment_magnitude NUMERIC(10, 6)
            )
            """
            await async_execute(create_table_query)
            log.info(f"Created table {table_name}")
            return True
        return True
    except Exception as e:
        log.error(f"Error checking/creating table {table_name}: {e}")
        return False

async def analyze_and_update_sentiment(
    table_name: str,
    text_column: str,
    id_column: str,
    batch_size: int = 100
) -> None:
    """
    Fetches records missing sentiment, analyzes text, and updates the table.

    Args:
        table_name: The name of the table to process (e.g., 'news_data', 'social_media_data').
        text_column: The name of the column containing the text to analyze.
        id_column: The name of the primary key column (or unique identifier).
        batch_size: Number of records to process in each batch.
    """
    log.info(f"Starting sentiment analysis for table: {table_name}")
    processed_count = 0
    
    # First check if table exists and create if needed
    try:
        table_exists = await check_and_create_table(table_name, text_column, id_column)
        if not table_exists:
            log.error(f"Cannot proceed with {table_name}: table doesn't exist and couldn't be created.")
            return
            
        # Check if required columns exist
        columns_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        """
        
        columns = await async_fetch(columns_query)
        if not columns:
            log.error(f"Could not retrieve column information for {table_name}")
            return
            
        column_names = [col['column_name'] for col in columns]
        
        # Check for text column and create if missing
        if text_column not in column_names:
            log.warning(f"Column {text_column} does not exist in table {table_name}. Adding it...")
            add_col_query = f"ALTER TABLE {table_name} ADD COLUMN {text_column} TEXT"
            await async_execute(add_col_query)
            
        # Check for ID column (cannot easily add primary key after creation)
        if id_column not in column_names:
            log.error(f"Column {id_column} does not exist in table {table_name} - this should be the primary key")
            return
            
        # Check if sentiment columns exist, create them if they don't
        sentiment_score_col = 'sentiment_score' 
        sentiment_magnitude_col = 'sentiment_magnitude'
        
        if sentiment_score_col not in column_names:
            log.info(f"Adding {sentiment_score_col} column to {table_name}")
            add_col_query = f"ALTER TABLE {table_name} ADD COLUMN {sentiment_score_col} NUMERIC(10, 6)"
            await async_execute(add_col_query)
            
        if sentiment_magnitude_col not in column_names:
            log.info(f"Adding {sentiment_magnitude_col} column to {table_name}")
            add_col_query = f"ALTER TABLE {table_name} ADD COLUMN {sentiment_magnitude_col} NUMERIC(10, 6)"
            await async_execute(add_col_query)
    except Exception as e:
        log.error(f"Error checking table structure for {table_name}: {e}")
        return
    
    # Now try to get records that need sentiment analysis
    try:
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        count_result = await async_fetch(count_query)
        total_records = count_result[0]['count'] if count_result else 0
        
        if total_records == 0:
            log.info(f"No records found in {table_name} to analyze.")
            return
            
        log.info(f"Found {total_records} total records in {table_name}")
        
        # Continue with normal analysis process for records without sentiment
        batch_attempt = 0
        while batch_attempt < 5:  # Limit attempts to prevent infinite loops
            log.info(f"Fetching batch of {batch_size} records from {table_name} needing analysis...")
            # Fetch records where sentiment score is NULL (or use another flag)
            query = f"""
            SELECT {id_column}, {text_column}
            FROM {table_name}
            WHERE sentiment_score IS NULL 
               OR sentiment_magnitude IS NULL
            LIMIT {batch_size}
            """
            try:
                records_to_analyze = await async_fetch(query)
            except Exception as e:
                column_error = f"column \"{text_column}\" does not exist" in str(e).lower()
                if column_error:
                    log.error(f"The column '{text_column}' does not exist in {table_name}, despite our check. This suggests a schema issue.")
                    return
                else:
                    log.error(f"Error fetching records from {table_name}: {e}")
                    batch_attempt += 1
                    await asyncio.sleep(5)
                    continue

            if not records_to_analyze:
                log.info(f"No more records found in {table_name} needing sentiment analysis.")
                break

            batch_attempt = 0  # Reset attempt counter on success
            log.info(f"Analyzing sentiment for {len(records_to_analyze)} records...")
            
            # Make sure the text is not None or empty
            valid_records = []
            for record in records_to_analyze:
                if record[text_column] and len(str(record[text_column]).strip()) > 0:
                    valid_records.append(record)
                else:
                    log.warning(f"Record with ID {record[id_column]} has empty or None text. Skipping.")
            
            if not valid_records:
                log.warning("No valid text content to analyze in this batch.")
                batch_attempt += 1
                await asyncio.sleep(3)
                continue
                
            ids = [record[id_column] for record in valid_records]
            texts = [str(record[text_column]) for record in valid_records]

            # Perform batch sentiment analysis
            try:
                sentiments = await analyze_texts_batch(texts)
            except Exception as e:
                log.error(f"Error during sentiment analysis: {e}")
                batch_attempt += 1
                await asyncio.sleep(5)
                continue

            if not sentiments or len(sentiments) != len(ids):
                log.warning(f"Sentiment analysis returned unexpected number of results ({len(sentiments)} vs {len(ids)}). Skipping batch.")
                batch_attempt += 1
                await asyncio.sleep(5)
                continue

            # Prepare data for update
            update_data = []
            for record_id, sentiment_result in zip(ids, sentiments):
                if sentiment_result:
                    update_data.append((
                        sentiment_result.get('score', 0.0),
                        sentiment_result.get('magnitude', 0.0),
                        record_id
                    ))
                else:
                    # Handle cases where analysis failed for a specific text
                    log.warning(f"Sentiment analysis failed for record ID {record_id} in {table_name}. Setting to neutral.")
                    update_data.append((0.0, 0.0, record_id))

            # Update the database
            if update_data:
                update_query = f"""
                UPDATE {table_name}
                SET 
                    sentiment_score = $1,
                    sentiment_magnitude = $2
                WHERE {id_column} = $3
                """
                try:
                    # Run updates asynchronously, one at a time
                    update_tasks = [async_execute(update_query, *row) for row in update_data]
                    await asyncio.gather(*update_tasks)
                    processed_count += len(update_data)
                    log.info(f"Updated sentiment for {len(update_data)} records in {table_name}. Total processed: {processed_count}")
                except Exception as e:
                    log.error(f"Error updating {table_name}: {e}")
                    batch_attempt += 1
                    await asyncio.sleep(10)  # Wait longer on DB error
                    continue
            
            # Small delay before fetching the next batch
            await asyncio.sleep(1)
    except Exception as e:
        log.error(f"Unexpected error during sentiment analysis for {table_name}: {e}")
        
    log.info(f"Finished sentiment analysis for table: {table_name}. Processed {processed_count} records.")

async def main():
    """Main function to run sentiment analysis on backfilled data."""
    log.info("Starting backfill sentiment analysis process...")

    # Analyze news data
    await analyze_and_update_sentiment(
        table_name='news_data', 
        text_column='text', 
        id_column='id'
    )

    # Analyze social media data
    await analyze_and_update_sentiment(
        table_name='social_media_data', 
        text_column='text_content', 
        id_column='id'
    )

    log.info("Backfill sentiment analysis process finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Sentiment analysis process interrupted by user.")
    except Exception as e:
        log.critical(f"An unhandled error occurred during sentiment analysis: {e}", exc_info=True)
        sys.exit(1)
