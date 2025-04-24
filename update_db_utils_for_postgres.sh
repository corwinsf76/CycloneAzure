#!/bin/bash
# update_db_utils_for_postgres.sh - Script to update database utilities for PostgreSQL

echo "=== Updating Database Utilities for PostgreSQL ==="

# Check for the database directories
if [ ! -d "database" ] || [ ! -d "cyclonev22/database" ]; then
    echo "Error: Could not find database directories"
    exit 1
fi

# Create backups of existing files
if [ -f database/db_utils.py ]; then
    echo "Creating backup of database/db_utils.py..."
    cp database/db_utils.py database/db_utils.py.mssql.bak
fi

if [ -f cyclonev22/database/db_utils.py ]; then
    echo "Creating backup of cyclonev22/database/db_utils.py..."
    cp cyclonev22/database/db_utils.py cyclonev22/database/db_utils.py.mssql.bak
fi

# Update PostgreSQL-specific utility functions
echo "Updating utility functions for PostgreSQL optimization..."

# Add PostgreSQL-specific imports and functions
cat <<'EOF' > database/postgres_utils.py
"""
PostgreSQL-specific database utility functions.
"""
import asyncpg
import logging
from typing import List, Dict, Any, Optional, Union
import config

log = logging.getLogger(__name__)

async def get_db_pool():
    """Get a connection pool to the PostgreSQL database."""
    try:
        return await asyncpg.create_pool(config.DATABASE_URL)
    except Exception as e:
        log.error(f"Error creating database pool: {e}")
        return None

async def execute_query(query: str, values: List[Any] = None) -> List[Dict[str, Any]]:
    """Execute a query and return the results as a list of dictionaries."""
    pool = await get_db_pool()
    if not pool:
        return []
    
    try:
        async with pool.acquire() as conn:
            if values:
                records = await conn.fetch(query, *values)
            else:
                records = await conn.fetch(query)
            
            # Convert records to dictionaries
            return [dict(record) for record in records]
    except Exception as e:
        log.error(f"Error executing query: {e}")
        return []
    finally:
        await pool.close()

async def execute_many(query: str, values_list: List[List[Any]]) -> bool:
    """Execute a query with multiple sets of values."""
    pool = await get_db_pool()
    if not pool:
        return False
    
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Prepare the statement
                stmt = await conn.prepare(query)
                # Execute for each set of values
                for values in values_list:
                    await stmt.fetch(*values)
            return True
    except Exception as e:
        log.error(f"Error executing bulk query: {e}")
        return False
    finally:
        await pool.close()

async def get_table_names() -> List[str]:
    """Get a list of table names in the database."""
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    """
    result = await execute_query(query)
    return [row['table_name'] for row in result]

async def get_column_info(table_name: str) -> List[Dict[str, Any]]:
    """Get column information for a table."""
    query = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = $1
    ORDER BY ordinal_position
    """
    return await execute_query(query, [table_name])

async def copy_data_from_csv(table_name: str, file_path: str, columns: List[str] = None) -> int:
    """
    Efficiently copy data from a CSV file to a table using PostgreSQL COPY.
    Returns the number of rows copied.
    """
    pool = await get_db_pool()
    if not pool:
        return 0
    
    try:
        async with pool.acquire() as conn:
            col_str = f"({','.join(columns)})" if columns else ""
            with open(file_path, 'r') as f:
                result = await conn.copy_to_table(
                    table_name, 
                    source=f, 
                    columns=columns
                )
            return result
    except Exception as e:
        log.error(f"Error copying data from CSV: {e}")
        return 0
    finally:
        await pool.close()

# Add more PostgreSQL-specific utility functions as needed
EOF

echo "PostgreSQL utility functions created in database/postgres_utils.py"

# Make the scripts executable
chmod +x update_db_utils_for_postgres.sh

echo "=== Database Utility Update Complete ==="
echo ""
echo "Next steps:"
echo "1. Update your code to import the new PostgreSQL utilities where needed"
echo "2. Test your application with the new database utilities"
echo "3. Consider updating the original db_utils.py files to use the new PostgreSQL functions"