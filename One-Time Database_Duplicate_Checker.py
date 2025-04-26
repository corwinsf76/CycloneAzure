#!/usr/bin/env python3
"""
One-Time Database Duplicate Checker (Azure Version)
"""

import asyncio
import asyncpg

async def check_duplicates():
    # Connect to your Azure PostgreSQL database
    conn = await asyncpg.connect(
        host="cyclonev2.postgres.database.azure.com",
        port=5432,
        database="postgres",
        user="Justin",
        password="Thomas12",
        ssl="require"  # Force SSL encryption
    )

    checks = [
        {
            "table": "price_data",
            "columns": ["symbol", "interval", "open_time"],
            "description": "Price data duplicates (symbol + interval + open_time)"
        },
        {
            "table": "news_data",
            "columns": ["url"],
            "description": "News article duplicates (url)"
        },
        {
            "table": "reddit_data",
            "columns": ["post_id"],
            "description": "Reddit post duplicates (post_id)"
        },
        {
            "table": "twitter_data",
            "columns": ["tweet_id"],
            "description": "Twitter post duplicates (tweet_id)"
        }
    ]

    for check in checks:
        print(f"\nChecking {check['description']}...\n")
        columns = ", ".join(check["columns"])
        query = f"""
            SELECT {columns}, COUNT(*) 
            FROM {check['table']} 
            GROUP BY {columns} 
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
        """
        try:
            rows = await conn.fetch(query)
            if not rows:
                print(f"✅ No duplicates found in {check['table']}.")
            else:
                print(f"⚠️ Found {len(rows)} duplicate entries in {check['table']}:")
                for row in rows:
                    print(dict(row))
        except Exception as e:
            print(f"Error checking {check['table']}: {e}")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_duplicates())
