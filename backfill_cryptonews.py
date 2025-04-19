# /backfill_cryptonews.py

import logging
import time
import datetime
import pytz
import pandas as pd
import json
from typing import List

# Setup project path for imports if running script directly
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # Assumes script is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # Use print before logging configured if needed
    # print(f"Added project root to sys.path: {project_root}")

# Import project modules
import config
from database import db_utils
from data_collection import cryptonews_client, binance_client # Need binance client to get symbols

# --- Configuration ---
# Basic logging config (can be enhanced)
logging.basicConfig(level=config.LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')
log = logging.getLogger(__name__)

# Define backfill period
BACKFILL_DAYS = config.get_env_variable("BACKFILL_DAYS_NEWS", 90, var_type=int) # Optional: Specific config var


# --- Backfill Function ---

def backfill_cryptonews_data(symbols: List[str], start_dt_utc: datetime.datetime, end_dt_utc: datetime.datetime):
    """Fetches historical news from CryptoNews API, handles pagination, and stores in DB."""
    log.info(f"Starting CryptoNews backfill for {len(symbols)} symbols from {start_dt_utc} to {end_dt_utc}...")
    if not db_utils.engine:
        log.critical("Database engine not configured. Cannot backfill CryptoNews data.")
        return
    if not config.CRYPTONEWS_API_TOKEN:
        log.critical("CRYPTONEWS_API_TOKEN not configured. Cannot backfill CryptoNews data.")
        return

    # --- Format date string for API ---
    # Use MMDDYYYY-MMDDYYYY format
    date_format = "%m%d%Y"
    date_str = f"{start_dt_utc.strftime(date_format)}-{end_dt_utc.strftime(date_format)}"
    log.info(f"Using date string for API: {date_str}")

    all_news_items_to_insert = []
    processed_urls = set() # Keep track of URLs processed in this run to avoid duplicates within the run

    # --- Process symbols ---
    # Get base symbols (like BTC, ETH) from the input list (which might be BTCUSDT etc.)
    base_symbols = list(set([s.replace('USDT', '').replace('BUSD','') for s in symbols]))
    if not base_symbols:
        log.warning("CryptoNews Backfill: No base symbols derived from input. Skipping.")
        return

    # Fetching per ticker seems required based on original script logic
    total_fetched_across_symbols = 0
    for i, symbol_base in enumerate(base_symbols):
        log.info(f"Fetching CryptoNews for {symbol_base} ({i+1}/{len(base_symbols)}), Date Range: {date_str}...")
        current_page = 1
        max_items_per_page = 100 # Fetch max items per page
        max_pages_per_symbol = 20 # Safety limit per symbol per run (adjust as needed)
        symbol_fetched_count = 0
        has_more = True # Assume there might be data initially

        while has_more and current_page <= max_pages_per_symbol:
            log.debug(f"Fetching page {current_page} for {symbol_base}...")
            try:
                # Use the specific historical fetch function from the client
                news_items_page, has_more_pages = cryptonews_client.fetch_historical_ticker_news(
                    tickers=[symbol_base], # Fetch for one symbol at a time
                    date_str=date_str,
                    items_per_page=max_items_per_page,
                    page=current_page
                )

                # Update loop control based on API response
                has_more = has_more_pages

                if not news_items_page:
                    log.info(f"No news items found for {symbol_base} on page {current_page}.")
                    # Stop pagination for this symbol if no items are returned
                    has_more = False # Force stop for this symbol
                else:
                    log.info(f"Fetched {len(news_items_page)} news items on page {current_page} for {symbol_base}.")
                    page_insert_candidates = []
                    for item in news_items_page:
                        url = item.get('url')
                        # Client should provide UTC datetime objects directly
                        published_dt = item.get('published_at')

                        # Filter by date again locally just in case API includes boundary items incorrectly
                        if published_dt and start_dt_utc <= published_dt <= end_dt_utc:
                           # Also check if URL has already been added in this run (avoids dupes from API glitches)
                           if url and url not in processed_urls:
                                processed_urls.add(url)
                                record = {
                                    'source_api': item.get('source_api', 'cryptonews'),
                                    'source_publisher': item.get('source_publisher'),
                                    'article_id': item.get('article_id'),
                                    'title': item.get('title'),
                                    'text_content': item.get('text_content'),
                                    'url': url,
                                    'published_at': published_dt,
                                    'tickers_mentioned': item.get('tickers_mentioned') # Already list or JSON string
                                }
                                # Ensure tickers_mentioned is serializable if DB requires string
                                if isinstance(record['tickers_mentioned'], list) and not (db_utils.JSON_TYPE.__name__ == 'JSON' or db_utils.JSON_TYPE.__name__ == 'JSONB'):
                                    record['tickers_mentioned'] = json.dumps(record['tickers_mentioned'])
                                page_insert_candidates.append(record)
                           elif url in processed_urls:
                               log.debug(f"Skipping duplicate URL within this run: {url}")
                           else:
                               log.debug(f"Skipping item for {symbol_base} due to missing URL or publish date.")
                        else:
                            # Log if item is outside date range but fetched anyway
                            if published_dt:
                                log.debug(f"Skipping item for {symbol_base} with publish date {published_dt} outside requested range {start_dt_utc} - {end_dt_utc}")


                    if page_insert_candidates:
                        all_news_items_to_insert.extend(page_insert_candidates)
                        symbol_fetched_count += len(page_insert_candidates)
                    else:
                        log.debug(f"No valid news items prepared for insertion on page {current_page} for {symbol_base} after filtering.")

                # Stop if API indicates no more pages
                if not has_more:
                    log.info(f"Reached last page for {symbol_base} according to API or lack of results.")
                    break # Exit while loop for this symbol

                current_page += 1
                # Add delay between pages for the same symbol
                time.sleep(1.5) # Be respectful of API limits

            except Exception as e:
                log.error(f"Error fetching/processing CryptoNews page {current_page} for {symbol_base}: {e}", exc_info=True)
                has_more = False # Stop pagination for this symbol on error
                time.sleep(5) # Wait longer after error
                break # Exit while loop for this symbol
        # --- End of while loop for pages ---

        total_fetched_across_symbols += symbol_fetched_count
        log.info(f"Fetched {symbol_fetched_count} total valid news items for {symbol_base} across all pages.")
        # Add delay between symbols
        time.sleep(2.5)

    # --- End of for loop for symbols ---

    # --- Store collected data ---
    if all_news_items_to_insert:
        log.info(f"Attempting to store {len(all_news_items_to_insert)} unique news items from backfill...")
        try:
            # Use bulk insert with URL as the unique identifier
            inserted_count = db_utils.bulk_insert_data(all_news_items_to_insert, db_utils.news_data, unique_column='url')
            log.info(f"Database insert call finished. Inserted count (if supported by DB driver): {inserted_count}. Attempted: {len(all_news_items_to_insert)}")
        except Exception as db_err:
            log.error(f"Error storing news data to DB: {db_err}", exc_info=True)
    else:
        log.info("No new news items collected during backfill to store.")

    log.info(f"--- CryptoNews Backfill Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    log.info(f"--- Starting CryptoNews Backfill Script ({BACKFILL_DAYS} days) ---")

    # --- Determine Date Range ---
    end_date = datetime.datetime.now(pytz.utc)
    start_date = end_date - pd.Timedelta(days=BACKFILL_DAYS)
    log.info(f"Backfill Range: {start_date} -> {end_date}")

    # --- Initialize DB ---
    if not db_utils.engine:
         log.critical("DB Engine not available. Exiting.")
         sys.exit(1)
    if not db_utils.init_db():
        log.critical("Database initialization failed. Exiting backfill.")
        sys.exit(1)
    log.info("Database schema initialized successfully.")

    # --- Get Symbols ---
    log.info("Getting target symbols from Binance...")
    try:
        target_symbols = binance_client.get_target_symbols()
    except Exception as e:
        log.critical(f"Failed to get target symbols from Binance: {e}. Exiting.")
        sys.exit(1)

    if not target_symbols:
        log.warning("No target symbols found from Binance. Nothing to backfill news for.")
        sys.exit(0)
    log.info(f"Found {len(target_symbols)} target symbols (e.g., {target_symbols[:5]}...).")

    # --- Run Backfill ---
    try:
        backfill_cryptonews_data(target_symbols, start_date, end_date)
    except Exception as e:
        log.critical(f"An unexpected error occurred during the CryptoNews backfill process: {e}", exc_info=True)

    log.info("--- CryptoNews Backfill Script Finished ---")
    print("\nCryptoNews backfill process complete. Check logs for details and errors.")