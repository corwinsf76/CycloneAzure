#!/usr/bin/env python3
"""
Backfill Twitter data for sub-$1 USDT coins using Twitter Official API v2 and Tweepy.
"""

import asyncio
import logging

from data_collection.twitter_client import search_recent_tweets
from data_collection.binance_client import _send_request
from database.db_utils import get_db_pool

log = logging.getLogger(__name__)

async def fetch_sub_usd_symbols():
    """Fetch all USDT pairs under $1 from Binance."""
    data = _send_request("GET", "/api/v3/ticker/price")
    symbols = [item['symbol'] for item in data if item['symbol'].endswith("USDT") and float(item['price']) < 1]
    return symbols

async def backfill_twitter_data():
    pool = await get_db_pool()

    symbols = await fetch_sub_usd_symbols()
    log.info(f"Sub-$1 USDT Symbols for Twitter monitoring: {symbols}")

    async with pool.acquire() as conn:
        for symbol in symbols:
            cashtag = f"${symbol.replace('USDT', '')}"

            try:
                log.info(f"Searching tweets for {cashtag}...")

                tweets = search_recent_tweets(cashtag, max_results=100)

                for tweet in tweets:
                    tweet_id = tweet.id
                    text = tweet.text
                    author_id = tweet.author_id
                    created_at = tweet.created_at
                    public_metrics = tweet.public_metrics

                    await conn.execute(
                        """
                        INSERT INTO twitter_data (
                            tweet_id, author_id, text, created_at, public_metrics, fetched_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, NOW()
                        )
                        ON CONFLICT (tweet_id)
                        DO NOTHING;
                        """,
                        tweet_id, author_id, text, created_at, public_metrics
                    )

                log.info(f"Inserted {len(tweets)} tweets for {cashtag}")

                await asyncio.sleep(1)

            except Exception as e:
                log.error(f"Error backfilling tweets for {cashtag}: {e}")

    await pool.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    asyncio.run(backfill_twitter_data())
