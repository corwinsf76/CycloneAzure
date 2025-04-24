#!/usr/bin/env python3
"""
Database Migration Script for Cyclone v2

This script helps migrate data from SQL Server to PostgreSQL.
It performs the following tasks:
1. Extracts the schema from SQL Server
2. Creates equivalent tables in PostgreSQL
3. Transfers data from SQL Server to PostgreSQL
4. Validates the migration

Usage:
python migrate_database.py [--mode=schema|data|validate|clear|all]
"""

import os
import sys
import argparse
import logging
import re
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.schema import CreateTable
import asyncio
import asyncpg
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Command line arguments
parser = argparse.ArgumentParser(description="Migrate database from SQL Server to PostgreSQL")
parser.add_argument("--mode", choices=["schema", "data", "validate", "clear", "all"], 
                    default="all", help="Migration mode")
parser.add_argument("--source-url", help="Source database URL (SQL Server)")
parser.add_argument("--target-url", help="Target database URL (PostgreSQL)")
parser.add_argument("--batch-size", type=int, default=5000, 
                    help="Batch size for data migration")
parser.add_argument("--tables", help="Comma-separated list of tables to migrate (default: all)")
args = parser.parse_args()

# Get database URLs from arguments or environment
SOURCE_DB_URL = args.source_url or os.environ.get('SOURCE_DATABASE_URL')
TARGET_DB_URL = args.target_url or os.environ.get('DATABASE_URL')  # Use current DATABASE_URL as target

if not SOURCE_DB_URL:
    # Try to load from backup file
    backup_env_path = '.env.mssql.backup'
    if os.path.exists(backup_env_path):
        with open(backup_env_path, 'r') as f:
            for line in f:
                if line.startswith('DATABASE_URL='):
                    SOURCE_DB_URL = line.split('=', 1)[1].strip()
                    break

if not SOURCE_DB_URL or not TARGET_DB_URL:
    log.error("Source or target database URL not provided.")
    log.error("Set SOURCE_DATABASE_URL and DATABASE_URL in .env or provide --source-url and --target-url arguments.")
    sys.exit(1)

# Confirm URLs are of the correct type
if not SOURCE_DB_URL.startswith('mssql'):
    log.warning(f"Source URL doesn't seem to be SQL Server: {SOURCE_DB_URL[:10]}...")
if not TARGET_DB_URL.startswith('postgresql'):
    log.warning(f"Target URL doesn't seem to be PostgreSQL: {TARGET_DB_URL[:10]}...")

# Get table list if specified
table_list = None
if args.tables:
    table_list = [t.strip() for t in args.tables.split(',')]

# Create database engines
try:
    source_engine = create_engine(SOURCE_DB_URL)
    target_engine = create_engine(TARGET_DB_URL)
    log.info("Database engines created successfully")
    
    # Log database types
    source_dialect = source_engine.dialect.name
    target_dialect = target_engine.dialect.name
    log.info(f"Source database: {source_dialect}")
    log.info(f"Target database: {target_dialect}")
except Exception as e:
    log.error(f"Error creating database engines: {e}")
    traceback.print_exc()
    sys.exit(1)

# Type mapping from SQL Server to PostgreSQL
def map_column_type(column_type):
    """Map SQL Server column types to PostgreSQL"""
    type_map = {
        'NVARCHAR': 'VARCHAR',
        'DATETIME': 'TIMESTAMP WITH TIME ZONE',
        'DATETIME2': 'TIMESTAMP WITH TIME ZONE',
        'DATETIMEOFFSET': 'TIMESTAMP WITH TIME ZONE',
        'BIT': 'BOOLEAN',
        'MONEY': 'NUMERIC',
        'UNIQUEIDENTIFIER': 'UUID',
    }
    
    for sql_type, pg_type in type_map.items():
        if sql_type in str(column_type).upper():
            return column_type.replace(sql_type, pg_type)
    return column_type

def migrate_schema():
    """Migrate schema from SQL Server to PostgreSQL"""
    log.info("Starting schema migration...")
    
    source_inspector = inspect(source_engine)
    target_inspector = inspect(target_engine)
    
    # Get tables from source
    source_tables = source_inspector.get_table_names()
    log.info(f"Found {len(source_tables)} tables in source database")
    
    # Filter tables if list provided
    if table_list:
        source_tables = [t for t in source_tables if t in table_list]
        log.info(f"Filtered to {len(source_tables)} tables: {', '.join(source_tables)}")
    
    # Get existing tables in target
    target_tables = target_inspector.get_table_names()
    
    # Create metadata for source database
    source_metadata = MetaData()
    source_metadata.reflect(bind=source_engine)
    
    # Process each table
    for table_name in source_tables:
        if table_name in target_tables:
            log.info(f"Table {table_name} already exists in target database, skipping schema migration")
            continue
        
        log.info(f"Migrating schema for table: {table_name}")
        
        # Get table from source metadata
        if table_name not in source_metadata.tables:
            log.warning(f"Table {table_name} not found in source metadata, skipping")
            continue
            
        source_table = source_metadata.tables[table_name]
        
        try:
            # Get create statement and modify for PostgreSQL
            create_stmt = str(CreateTable(source_table).compile(dialect=source_engine.dialect))
            
            # Map column types
            for column in source_table.columns:
                old_type = str(column.type)
                new_type = map_column_type(column.type)
                if old_type != new_type:
                    create_stmt = create_stmt.replace(old_type, new_type)
            
            # Fix IDENTITY syntax for PostgreSQL
            create_stmt = re.sub(r'IDENTITY\(\d+,\s*\d+\)', 'GENERATED ALWAYS AS IDENTITY', create_stmt, flags=re.IGNORECASE)
            
            # Fix timestamp syntax
            create_stmt = create_stmt.replace('TIMESTAMP WITH TIME ZONEOFFSET', 'TIMESTAMP WITH TIME ZONE')
            
            # Fix default value for timestamps
            create_stmt = create_stmt.replace('DEFAULT (getdate())', "DEFAULT CURRENT_TIMESTAMP")
            
            # Execute the create statement
            with target_engine.connect() as conn:
                conn.execute(text(create_stmt))
                conn.commit()
            log.info(f"Successfully created table {table_name} in target database")
        except Exception as e:
            log.error(f"Error creating table {table_name}: {e}")
            traceback.print_exc()

def clear_target_tables():
    """Truncate all tables in the target database"""
    log.info("Clearing data from target tables...")
    target_inspector = inspect(target_engine)
    target_tables = target_inspector.get_table_names()

    # Filter tables if list provided
    tables_to_clear = table_list if table_list else target_tables
    
    cleared_count = 0
    skipped_count = 0
    error_count = 0

    if not tables_to_clear:
        log.info("No tables found in target to clear.")
        return True

    log.info(f"Attempting to clear {len(tables_to_clear)} tables: {', '.join(tables_to_clear)}")

    with target_engine.connect() as conn:
        transaction = conn.begin()
        try:
            # Temporarily disable foreign key constraints if possible (dialect specific)
            if target_engine.dialect.name == 'postgresql':
                conn.execute(text("SET session_replication_role = 'replica';"))
                log.info("Temporarily disabled foreign key checks (PostgreSQL).")

            for table_name in tables_to_clear:
                # Check if table exists before trying to truncate
                if table_name in target_tables:
                    try:
                        log.info(f"Clearing table: {table_name}")
                        # Use TRUNCATE for efficiency, CASCADE for foreign keys if needed
                        conn.execute(text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;'))
                        cleared_count += 1
                    except Exception as e:
                        log.error(f"Error clearing table {table_name}: {e}")
                        error_count += 1
                else:
                    log.warning(f"Table {table_name} not found in target database, skipping clear.")
                    skipped_count += 1
            
            # Re-enable foreign key constraints
            if target_engine.dialect.name == 'postgresql':
                 conn.execute(text("SET session_replication_role = 'origin';"))
                 log.info("Re-enabled foreign key checks (PostgreSQL).")

            transaction.commit()
            log.info(f"Target tables cleared: {cleared_count} cleared, {skipped_count} skipped, {error_count} errors.")
            return error_count == 0
        except Exception as e:
            log.error(f"Error during table clearing transaction: {e}")
            transaction.rollback()
            # Try to re-enable FKs even on error
            if target_engine.dialect.name == 'postgresql':
                 try:
                     conn.execute(text("SET session_replication_role = 'origin';"))
                 except Exception as fk_err:
                     log.error(f"Could not re-enable FK checks after error: {fk_err}")
            return False

def migrate_data():
    """Migrate data from SQL Server to PostgreSQL"""
    log.info("Starting data migration...")

    # Check if target tables already have data
    target_inspector = inspect(target_engine)
    target_tables = target_inspector.get_table_names()
    has_existing_data = False

    for table_name in target_tables:
        try:
            with target_engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                count = result.scalar()
                if count > 0:
                    log.warning(f"Table {table_name} already contains {count} rows")
                    has_existing_data = True
        except Exception as e:
            log.error(f"Error checking table {table_name}: {e}")
            return False

    if has_existing_data:
        user_input = input("Target database already contains data. Do you want to clear and remigrate? (y/N): ").lower()
        if user_input != 'y':
            log.info("Migration cancelled by user")
            return False
    
    # Clear target tables before migrating data to prevent duplicates
    if not clear_target_tables():
         log.error("Failed to clear target tables. Aborting data migration.")
         return False

    source_inspector = inspect(source_engine)
    
    # Get tables from source
    source_tables = source_inspector.get_table_names()
    
    # Filter tables if list provided
    if table_list:
        source_tables = [t for t in source_tables if t in table_list]
    
    # Process each table
    for table_name in source_tables:
        log.info(f"Migrating data for table: {table_name}")
        
        try:
            # Get row count
            with source_engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()
            
            log.info(f"Table {table_name} has {row_count} rows")
            
            # Skip if empty
            if row_count == 0:
                log.info(f"Table {table_name} is empty, skipping")
                continue
            
            # Migrate in batches
            batch_size = args.batch_size
            batches = (row_count + batch_size - 1) // batch_size  # Ceiling division
            
            for i in range(batches):
                offset = i * batch_size
                limit = min(batch_size, row_count - offset)
                
                log.info(f"Migrating batch {i+1}/{batches} (offset {offset}, limit {limit})")
                
                # Read batch from source - handle different SQL dialects
                if source_engine.dialect.name == 'mssql':
                    # SQL Server syntax
                    query = f"SELECT * FROM {table_name} ORDER BY (SELECT NULL) OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
                else:
                    # Generic SQL syntax
                    query = f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}"
                
                try:
                    df = pd.read_sql(query, source_engine)
                except Exception as e:
                    log.error(f"Error reading data from source: {e}")
                    # Try alternative syntax for older SQL Server
                    if source_engine.dialect.name == 'mssql':
                        log.info("Trying alternative SQL Server paging syntax...")
                        # Find a suitable ID column
                        with source_engine.connect() as conn:
                            # Get primary key or first column
                            pk_result = conn.execute(text(f"""
                                SELECT COLUMN_NAME
                                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                                WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + QUOTENAME(CONSTRAINT_NAME)), 'IsPrimaryKey') = 1
                                AND TABLE_NAME = '{table_name}'
                            """))
                            pk_cols = pk_result.fetchall()
                            
                            if pk_cols:
                                id_col = pk_cols[0][0]
                                log.info(f"Using primary key column: {id_col}")
                                
                                # Use TOP and WHERE for paging
                                query = f"""
                                SELECT TOP {limit} * FROM {table_name}
                                WHERE {id_col} NOT IN (
                                    SELECT TOP {offset} {id_col} FROM {table_name} ORDER BY {id_col}
                                )
                                ORDER BY {id_col}
                                """
                                df = pd.read_sql(query, source_engine)
                            else:
                                # No primary key, just use TOP
                                log.warning(f"No primary key found for {table_name}, using simple TOP query")
                                query = f"SELECT TOP {limit} * FROM {table_name}"
                                df = pd.read_sql(query, source_engine)
                
                # Handle data type conversions
                for col in df.select_dtypes(include=['datetime64']):
                    # Ensure timezone-aware datetimes for PostgreSQL
                    try:
                        df[col] = df[col].dt.tz_localize('UTC')
                    except:
                        # If already tz-aware or another error
                        pass
                
                # Replace any NaN/inf values
                df = df.replace([float('inf'), float('-inf')], None)
                
                # Write to target
                try:
                    # First validate no duplicates exist
                    with target_engine.connect() as conn:
                        count_before = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                    
                    df.to_sql(
                        table_name, 
                        target_engine, 
                        if_exists='append', 
                        index=False,
                        chunksize=1000
                    )

                    # Verify the correct number of rows were added
                    with target_engine.connect() as conn:
                        count_after = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                        rows_added = count_after - count_before
                        if rows_added != len(df):
                            log.error(f"Data duplication detected. Expected to add {len(df)} rows but added {rows_added}")
                            # Rollback the batch by deleting the most recently added rows
                            conn.execute(text(f'DELETE FROM "{table_name}" WHERE ctid IN (SELECT ctid FROM "{table_name}" ORDER BY ctid DESC LIMIT {rows_added})'))
                            conn.commit()
                            return False

                except Exception as e:
                    log.error(f"Error writing to target: {e}")
                    # Try a more conservative approach
                    log.info("Trying more conservative approach with explicit column handling...")
                    
                    # Handle column types more explicitly
                    for col in df.select_dtypes(include=['datetime64']):
                        df[col] = df[col].astype(str)
                    
                    df.to_sql(
                        table_name, 
                        target_engine, 
                        if_exists='append', 
                        index=False,
                        chunksize=100
                    )
                
                log.info(f"Migrated {len(df)} rows to {table_name}")
        
        except Exception as e:
            log.error(f"Error migrating data for table {table_name}: {e}")
            traceback.print_exc()
            return False

    log.info(f"Data migration finished for all tables.")
    return True

def validate_migration():
    """Validate the migration by comparing row counts"""
    log.info("Validating migration...")
    
    source_inspector = inspect(source_engine)
    
    # Get tables from source
    source_tables = source_inspector.get_table_names()
    
    # Filter tables if list provided
    if table_list:
        source_tables = [t for t in source_tables if t in table_list]
    
    validation_passed = True
    
    # Compare row counts for each table
    for table_name in source_tables:
        try:
            # Get source row count
            with source_engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                source_count = result.scalar()
            
            # Get target row count
            with target_engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                target_count = result.scalar()
            
            if source_count == target_count:
                log.info(f"Table {table_name}: Validation PASSED ({source_count} rows)")
            else:
                log.error(f"Table {table_name}: Validation FAILED - Source: {source_count} rows, Target: {target_count} rows")
                validation_passed = False
        
        except Exception as e:
            log.error(f"Error validating table {table_name}: {e}")
            validation_passed = False
    
    return validation_passed

async def test_connections():
    """Test connections to both databases"""
    log.info("Testing database connections...")
    
    # Test SQL Server connection
    try:
        with source_engine.connect() as conn:
            if source_engine.dialect.name == 'mssql':
                result = conn.execute(text("SELECT @@version"))
            else:
                result = conn.execute(text("SELECT 1"))
            version = result.scalar()
            log.info(f"SQL Server connection successful - version: {version[:50] if isinstance(version, str) else version}")
    except Exception as e:
        log.error(f"SQL Server connection failed: {e}")
        return False
    
    # Test PostgreSQL connection
    try:
        with target_engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            log.info(f"PostgreSQL connection successful - version: {version[:50] if isinstance(version, str) else version}")
    except Exception as e:
        log.error(f"PostgreSQL connection failed: {e}")
        return False
    
    # Test PostgreSQL async connection if it's PostgreSQL
    if target_engine.dialect.name == 'postgresql':
        try:
            conn = await asyncpg.connect(TARGET_DB_URL)
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            log.info(f"PostgreSQL async connection successful")
        except Exception as e:
            log.error(f"PostgreSQL async connection failed: {e}")
            log.warning("Continuing without async capability")
    
    return True

async def main():
    """Main function"""
    log.info("Starting database migration process")
    
    # Test connections
    connections_ok = await test_connections()
    if not connections_ok:
        log.error("Database connection tests failed, aborting migration")
        return 1
    
    # Determine which operations to perform
    mode = args.mode.lower()
    
    schema_success = True
    clear_success = True
    data_success = True
    validation_success = True

    if mode in ["schema", "all"]:
        schema_success = migrate_schema()
        if not schema_success:
            log.error("Schema migration failed.")

    if mode == "clear":
        clear_success = clear_target_tables()
        if not clear_success:
            log.error("Clearing target tables failed.")
            return 1

    if mode in ["data", "all"]:
        data_success = migrate_data()
        if not data_success:
            log.error("Data migration failed.")

    if mode in ["validate", "all"]:
        validation_success = validate_migration()
        if validation_success:
            log.info("Migration validation PASSED")
        else:
            log.error("Migration validation FAILED")
            return 1
    
    if schema_success and data_success and validation_success and clear_success:
        log.info("Database migration completed successfully")
        return 0
    else:
        log.error("Database migration process finished with errors.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)