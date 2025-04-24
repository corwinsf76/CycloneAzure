import asyncio
import asyncpg
import sys

async def test_connection():
    try:
        print(f"Connecting to: postgresql://Justin:******@cyclonev2.postgres.database.azure.com:5432/postgres?sslmode=require")
        
        conn = await asyncpg.connect(
            host="cyclonev2.postgres.database.azure.com",
            port=5432,
            user="Justin",
            password="Thomas12",
            database="postgres",
            ssl=True
        )
        
        # Test simple query
        version = await conn.fetchval('SELECT version()')
        print(f"Connected successfully!")
        print(f"PostgreSQL version: {version}")
        
        await conn.close()
        return True
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    sys.exit(0 if result else 1)
