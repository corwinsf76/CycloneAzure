#!/usr/bin/env python3
"""
Database Connection Test Script

This script tests the database connection using SQLAlchemy and pyodbc.
It will attempt to connect to the database defined in config.py
and confirm that the connection is working.
"""

import logging
import sys
import os
import traceback
from pprint import pprint
import subprocess

# Configure direct output to stdout
print("=== Database Connection Test Script ===")
print(f"Python version: {sys.version}")

# Check if required packages are installed
try:
    print("\n--- Checking Required Packages ---")
    import pkg_resources
    required_packages = ['sqlalchemy', 'pyodbc', 'asyncpg']
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for package in required_packages:
        if package in installed_packages:
            print(f"✓ {package} {installed_packages[package]} is installed")
        else:
            print(f"✗ {package} is NOT installed")
    
    # Try to import required modules
    print("\n--- Testing Imports ---")
    import sqlalchemy
    print(f"✓ SQLAlchemy version: {sqlalchemy.__version__}")
    
    import pyodbc
    print(f"✓ pyodbc version: {pyodbc.version}")
    
    # Check ODBC driver availability
    print("\n--- ODBC Driver Information ---")
    try:
        drivers = pyodbc.drivers()
        print(f"Available ODBC drivers: {drivers}")
        
        if 'ODBC Driver 18 for SQL Server' not in drivers:
            print("✗ WARNING: 'ODBC Driver 18 for SQL Server' not found in pyodbc drivers list!")
    except Exception as e:
        print(f"Error checking ODBC drivers: {e}")
except ImportError as e:
    print(f"Failed to import required packages: {e}")
    print("Please install the required packages with:")
    print("pip install sqlalchemy pyodbc asyncpg")
    sys.exit(1)

# Add project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"\nAdded current directory to sys.path: {current_dir}")

# Import config
try:
    print("\n--- Loading Configuration ---")
    import config
    
    # Check DATABASE_URL
    if not hasattr(config, 'DATABASE_URL') or not config.DATABASE_URL:
        print("ERROR: DATABASE_URL is not configured in config.py or environment")
        print("Available config values:")
        print([attr for attr in dir(config) if not attr.startswith('_') and not callable(getattr(config, attr))])
        sys.exit(1)
    
    # Mask the password for security
    url_parts = config.DATABASE_URL.split('@')
    if len(url_parts) > 1:
        user_part = url_parts[0].split(':')
        if len(user_part) > 2:
            masked_url = f"{user_part[0]}:****@{url_parts[1]}"
            print(f"Database URL: {masked_url}")
        else:
            print(f"Database URL: {url_parts[0].split(':')[0]}:****@{url_parts[1]}")
    else:
        print(f"Database URL format could not be parsed for masking")
        
except ImportError as e:
    print(f"Failed to import config module: {e}")
    sys.exit(1)

# Test direct connection with pyodbc
print("\n--- Testing Direct pyodbc Connection ---")
try:
    # Parse connection string to extract components
    conn_str = config.DATABASE_URL.replace('mssql+pyodbc://', '')
    
    # Split username:password@server/database
    auth_server, rest = conn_str.split('@', 1)
    username, password = auth_server.split(':', 1)
    
    # Split server/database?params
    server_part, params = rest.split('?', 1) if '?' in rest else (rest, '')
    server, database = server_part.split('/', 1) if '/' in server_part else (server_part, '')
    
    # Extract port if present
    if ':' in server:
        server, port = server.split(':', 1)
    else:
        port = '1433'  # Default SQL Server port
    
    # Build direct ODBC connection string
    driver = 'ODBC Driver 18 for SQL Server'
    direct_conn_str = (
        f'DRIVER={{{driver}}};'
        f'SERVER={server},{port};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
        f'Encrypt=yes;'
        f'TrustServerCertificate=no;'
        f'Connection Timeout=30;'
    )
    
    print(f"Attempting to connect with pyodbc directly...")
    print(f"Server: {server}")
    print(f"Database: {database}")
    print(f"Driver: {driver}")
    
    conn = pyodbc.connect(direct_conn_str)
    cursor = conn.cursor()
    
    print("✓ Direct pyodbc connection successful!")
    
    # Test query
    cursor.execute("SELECT @@version")
    version = cursor.fetchone()[0]
    print(f"SQL Server version: {version}")
    
    # Get table list
    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
    tables = cursor.fetchall()
    print("\nTables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"✗ Direct pyodbc connection failed: {e}")
    traceback.print_exc()

# Test connection with SQLAlchemy
print("\n--- Testing SQLAlchemy Connection ---")
try:
    from sqlalchemy import create_engine, inspect
    
    engine = create_engine(config.DATABASE_URL, echo=False)
    print("Engine created successfully")
    
    with engine.connect() as connection:
        print("✓ SQLAlchemy connection successful!")
        
        # Inspect tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\nFound {len(tables)} tables in the database:")
        for table in tables:
            print(f"  - {table}")
    
except Exception as e:
    print(f"✗ SQLAlchemy connection failed: {e}")
    traceback.print_exc()

print("\n--- Testing Database Utils Module ---")
try:
    # Import database utils
    from database import db_utils
    
    # Check if engine was created in module
    if not db_utils.engine:
        print("✗ db_utils engine not initialized")
    else:
        print("✓ db_utils engine initialized")
        
        # Test connection
        with db_utils.engine.connect() as connection:
            print("✓ db_utils engine connection successful")
            
            # Test schema initialization
            if db_utils.init_db():
                print("✓ Database schema initialization successful")
            else:
                print("✗ Database schema initialization failed")
                
except ImportError as e:
    print(f"✗ Failed to import database.db_utils module: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"✗ Error working with db_utils module: {e}")
    traceback.print_exc()

print("\n=== Database Connection Test Complete ===")