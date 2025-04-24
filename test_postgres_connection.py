import psycopg2
import sys

def test_connection():
    try:
        conn_str = "postgresql://Justin:Thomas12@cyclonev2.postgres.database.azure.com:5432/postgres?sslmode=require"
        print(f"Connecting to: postgresql://Justin:******@cyclonev2.postgres.database.azure.com:5432/postgres?sslmode=require")
        
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
