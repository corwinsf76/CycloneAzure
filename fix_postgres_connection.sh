#!/bin/bash
# Script to fix Azure PostgreSQL connection for Cyclone v2

echo "===== Azure PostgreSQL Connection Fixer ====="
echo "This script will update your config.py with the correct database connection string."

# Define database connection parameters
SERVER="cyclonev2.postgres.database.azure.com"
PORT=5432
DATABASE="postgres"
USERNAME="Justin"  # Using the correct username from your successful psql command
PASSWORD="Thomas12"  # Corrected password without spaces

# Generate the correct PostgreSQL connection URL
CONNECTION_STRING="postgresql://${USERNAME}:${PASSWORD}@${SERVER}:${PORT}/${DATABASE}?sslmode=require"

# Create a test file to verify connection with psycopg2
cat > test_postgres_connection.py << EOF
import psycopg2
import sys

def test_connection():
    try:
        conn_str = "${CONNECTION_STRING}"
        print(f"Connecting to: postgresql://${USERNAME}:******@${SERVER}:${PORT}/${DATABASE}?sslmode=require")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(conn_str)
        
        # Test simple query
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]
        
        print(f"Connected successfully!")
        print(f"PostgreSQL version: {version}")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    result = test_connection()
    sys.exit(0 if result else 1)
EOF

# Create a test file for asyncpg (which is used by your dashboard)
cat > test_asyncpg_connection.py << EOF
import asyncio
import asyncpg
import sys

async def test_connection():
    try:
        print(f"Connecting to: postgresql://${USERNAME}:******@${SERVER}:${PORT}/${DATABASE}?sslmode=require")
        
        conn = await asyncpg.connect(
            host="${SERVER}",
            port=${PORT},
            user="${USERNAME}",
            password="${PASSWORD}",
            database="${DATABASE}",
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
EOF

# Backup the original config.py
cp config.py config.py.bak
echo "✓ Created backup of config.py as config.py.bak"

# Update config.py with new connection string
sed -i "s|DATABASE_URL = .*|DATABASE_URL = \"${CONNECTION_STRING}\"|" config.py
echo "✓ Updated DATABASE_URL in config.py"

# Update the db_utils.py file to ensure correct connection string handling
cat > update_db_utils.py << EOF
import os
import sys
import re

# Path to db_utils.py
db_utils_path = os.path.join('database', 'db_utils.py')

if not os.path.exists(db_utils_path):
    print(f"Error: Could not find {db_utils_path}")
    sys.exit(1)

# Backup the file
os.system(f"cp {db_utils_path} {db_utils_path}.bak")

# Read the file
with open(db_utils_path, 'r') as f:
    content = f.read()

# Check if the file uses asyncpg
uses_asyncpg = 'asyncpg' in content

if uses_asyncpg:
    print("The db_utils.py file uses asyncpg for database connections.")
    
    # Update the get_pool function if needed
    pool_pattern = r'async def get_pool\(\).*?global _pool.*?if _pool is None.*?try:.*?_pool = await asyncpg\.create_pool\((.*?)\)'
    pool_match = re.search(pool_pattern, content, re.DOTALL)
    
    if pool_match:
        print("Found get_pool function, updating connection parameters...")
        
        # Replace with explicit connection parameters
        new_pool_code = '''async def get_pool():
    """Get or create the database connection pool"""
    global _pool
    if _pool is None:
        try:
            # Parse DATABASE_URL to get components
            url = DATABASE_URL
            if '?sslmode=require' in url:
                url = url.replace('?sslmode=require', '')
            
            if '@' in url and ':' in url:
                user_pass = url.split('@')[0].split('://')[1]
                user = user_pass.split(':')[0]
                password = user_pass.split(':')[1]
                
                host_port_db = url.split('@')[1]
                host = host_port_db.split(':')[0]
                port_db = host_port_db.split(':')[1]
                port = port_db.split('/')[0]
                database = port_db.split('/')[1]
                
                _pool = await asyncpg.create_pool(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    ssl=True
                )
            else:
                # Fallback to direct url
                _pool = await asyncpg.create_pool(DATABASE_URL)
        except Exception as e:
            log.error(f"Error creating connection pool: {e}")
            return None
    return _pool'''
        
        content = re.sub(pool_pattern, new_pool_code, content, flags=re.DOTALL)
        
        # Write updated content
        with open(db_utils_path, 'w') as f:
            f.write(content)
        
        print(f"✓ Updated {db_utils_path} to handle sslmode and password with spaces")
    else:
        print("Could not find get_pool function to update")

print("Database utility files updated successfully")
EOF

# Run the update script
python update_db_utils.py

echo ""
echo "===== Next Steps ====="
echo "1. Test the connection with: python test_postgres_connection.py"
echo "2. If that works, test the asyncpg connection: python test_asyncpg_connection.py"
echo "3. Then try running the dashboard: python dashboard/launch_dashboard.py"
echo ""
echo "If you still have issues, you may need to update your Azure PostgreSQL firewall rules"
echo "to allow your current IP address by running: ./update_azure_postgres_firewall.sh"
echo ""

# Make the script executable
chmod +x fix_postgres_connection.sh