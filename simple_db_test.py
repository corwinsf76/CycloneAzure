#!/usr/bin/env python3
"""
Simple Database Test Script

Tests basic database connectivity without depending on the full config system.
"""

import os
import sys
import sqlalchemy
import pyodbc
from sqlalchemy import create_engine, text

print("=== Simple Database Connection Test ===")
print(f"Python version: {sys.version}")
print(f"SQLAlchemy version: {sqlalchemy.__version__}")
print(f"pyodbc version: {pyodbc.version}")

# Get database URL from environment or use hardcoded value from test file
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    # Hardcoded value from test file (with password masked)
    print("DATABASE_URL not found in environment, using hardcoded value")
    DATABASE_URL = "mssql+pyodbc://justinlaughlin:****@cyclonev2master.database.windows.net:1433/cyclonev2database?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"

print(f"Using Database URL: {DATABASE_URL.replace(':****@', ':PASSWORD@') if ':' in DATABASE_URL else DATABASE_URL}")

try:
    print("\nAttempting to create SQLAlchemy engine...")
    engine = create_engine(DATABASE_URL)
    print("Engine created successfully!")
    
    print("\nAttempting to connect to database...")
    with engine.connect() as connection:
        print("Connection established successfully!")
        
        # Test a simple query
        print("\nExecuting test query...")
        result = connection.execute(text("SELECT @@version"))
        version = result.scalar()
        print(f"SQL Server version: {version}")
        
        # List tables
        print("\nListing database tables...")
        result = connection.execute(text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"))
        tables = [row[0] for row in result]
        if tables:
            print(f"Found {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
        else:
            print("No tables found in database.")
    
    print("\nDatabase connection test SUCCESSFUL!")
    sys.exit(0)
except Exception as e:
    print(f"\nERROR: Database connection failed: {e}")
    print("\nDetailed error information:")
    import traceback
    traceback.print_exc()
    print("\nDatabase connection test FAILED!")
    sys.exit(1)