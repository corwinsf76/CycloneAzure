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
