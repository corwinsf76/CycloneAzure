import asyncio
import asyncpg
from datetime import datetime, timedelta
import pandas as pd

class LowValueAnalyzer:
    def __init__(self, pool):
        self.pool = pool
        self.max_price = 1.0  # $1 threshold

    async def get_low_value_coins(self):
        async with self.pool.acquire() as conn:
            query = """
            SELECT DISTINCT coin_id, price, market_cap, volume_24h
            FROM coin_metrics
            WHERE price <= $1
            AND timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY market_cap DESC
            LIMIT 50
            """
            return await conn.fetch(query, self.max_price)

    async def get_sentiment_history(self, coin_id):
        async with self.pool.acquire() as conn:
            query = """
            SELECT timestamp, sentiment_score
            FROM coin_sentiment
            WHERE coin_id = $1
            AND timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY timestamp ASC
            """
            return await conn.fetch(query, coin_id)

    async def get_portfolio_holdings(self):
        async with self.pool.acquire() as conn:
            query = """
            SELECT 
                h.coin_id,
                h.quantity,
                cm.price,
                h.quantity * cm.price as current_value
            FROM holdings h
            JOIN coin_metrics cm ON h.coin_id = cm.coin_id
            WHERE cm.timestamp >= NOW() - INTERVAL '5 minutes'
            """
            return await conn.fetch(query)
