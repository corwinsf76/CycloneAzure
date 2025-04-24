#!/bin/bash

# Script to fix Azure PostgreSQL connection issues
# This script helps diagnose and fix common issues with Azure PostgreSQL connections

echo "===== Azure PostgreSQL Connection Fixer ====="
echo "This script will help diagnose and fix common connection issues."

# Get the current IP address
CURRENT_IP=$(curl -s https://api.ipify.org)
echo "Your current public IP address: $CURRENT_IP"

# Check for Azure CLI installation
if command -v az >/dev/null 2>&1; then
    echo "✓ Azure CLI is installed"
    # Check if logged in
    ACCOUNT_STATUS=$(az account show 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "✓ You are logged in to Azure CLI"
        SUBSCRIPTION=$(az account show --query name -o tsv)
        echo "  Current subscription: $SUBSCRIPTION"
    else
        echo "✗ You need to log in to Azure CLI"
        echo "  Run: az login"
        echo "  Then run this script again"
        exit 1
    fi
else
    echo "✗ Azure CLI is not installed"
    echo "  To install, follow instructions at: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    echo "  Then run this script again"
    exit 1
fi

# Extract server name from DATABASE_URL in config.py
SERVER_NAME=$(grep -o "postgresql://[^@]*@[^:]*" config.py | cut -d@ -f2)
if [ -z "$SERVER_NAME" ]; then
    echo "✗ Could not extract server name from config.py"
    echo "  Please enter your Azure PostgreSQL server name (e.g., cyclonev2.postgres.database.azure.com):"
    read SERVER_NAME
fi

echo "Server name: $SERVER_NAME"

# Extract server resource group
echo "Searching for PostgreSQL server in your Azure subscriptions..."
RESOURCE_GROUP=$(az postgres server list --query "[?fullyQualifiedDomainName=='$SERVER_NAME'].resourceGroup" -o tsv)

if [ -z "$RESOURCE_GROUP" ]; then
    echo "✗ Could not find PostgreSQL server in your subscriptions"
    echo "  Please enter the resource group name for your PostgreSQL server:"
    read RESOURCE_GROUP
fi

echo "Resource group: $RESOURCE_GROUP"

# Get server name without domain
SHORT_SERVER_NAME=$(echo $SERVER_NAME | cut -d. -f1)
echo "Short server name: $SHORT_SERVER_NAME"

# Update firewall rule
echo "Updating firewall rules to allow your current IP ($CURRENT_IP)..."
az postgres server firewall-rule create \
    --resource-group $RESOURCE_GROUP \
    --server-name $SHORT_SERVER_NAME \
    --name "AllowMyIP_$(date +%Y%m%d)" \
    --start-ip-address $CURRENT_IP \
    --end-ip-address $CURRENT_IP

if [ $? -eq 0 ]; then
    echo "✓ Firewall rule added successfully"
else
    echo "✗ Failed to add firewall rule"
    echo "  You may need to add it manually in Azure Portal"
fi

# Check current database user
USER_EMAIL=$(grep -o "postgresql://[^:]*" config.py | cut -d/ -f3)
echo "Current database user in config.py: $USER_EMAIL"
echo ""
echo "===== Connection String Troubleshooting ====="
echo "Your current connection string might be incorrect."
echo "For Azure PostgreSQL, the username format should be:"
echo "username@servername (where servername is the short name before .postgres.database.azure.com)"
echo ""
echo "Your connection string should look like:"
echo "postgresql://username@servername:password@servername.postgres.database.azure.com:5432/dbname"
echo ""
echo "Would you like to update your connection string in config.py? (y/n)"
read UPDATE_CONFIG

if [[ $UPDATE_CONFIG == "y" || $UPDATE_CONFIG == "Y" ]]; then
    echo "Enter your PostgreSQL username (without @servername):"
    read DB_USER
    
    echo "Enter your PostgreSQL password:"
    read -s DB_PASSWORD
    
    echo "Enter your database name (default is 'postgres'):"
    read DB_NAME
    if [ -z "$DB_NAME" ]; then
        DB_NAME="postgres"
    fi
    
    # Create the new connection string
    NEW_CONNECTION_STRING="postgresql://${DB_USER}@${SHORT_SERVER_NAME}:${DB_PASSWORD}@${SERVER_NAME}:5432/${DB_NAME}"
    
    # Backup config.py
    cp config.py config.py.bak
    
    # Update config.py
    sed -i "s|DATABASE_URL = .*|DATABASE_URL = \"${NEW_CONNECTION_STRING}\"|" config.py
    
    echo "✓ Updated connection string in config.py (backed up as config.py.bak)"
    
    # Check if using sqlalchemy or asyncpg in database/db_utils.py
    if grep -q "asyncpg" database/db_utils.py; then
        echo "Your project uses asyncpg for database connections."
        echo "For asyncpg, the connection string format should be:"
        echo "postgresql://username:password@servername.postgres.database.azure.com:5432/dbname"
        echo ""
        echo "Would you like to create a test file to verify your connection? (y/n)"
        read CREATE_TEST
        
        if [[ $CREATE_TEST == "y" || $CREATE_TEST == "Y" ]]; then
            cat > test_asyncpg_connection.py << EOF
import asyncio
import asyncpg
import sys

async def test_connection():
    try:
        # Extract username and password from the SQLAlchemy URL format
        conn_str = "${NEW_CONNECTION_STRING}"
        parts = conn_str.replace('postgresql://', '').split('@')
        user_part = parts[0].split(':')
        username = user_part[0]
        password = user_part[1]
        host_part = parts[1].split(':')
        host = host_part[0]
        port_db = host_part[1].split('/')
        port = port_db[0]
        db = port_db[1]
        
        # Format for asyncpg
        asyncpg_conn_str = f"postgresql://{username}:{password}@{host}:{port}/{db}"
        
        print(f"Connecting to: postgresql://{username}:******@{host}:{port}/{db}")
        conn = await asyncpg.connect(asyncpg_conn_str)
        
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
            echo "✓ Created test_asyncpg_connection.py"
            echo "  Run it with: python test_asyncpg_connection.py"
        fi
    fi
fi

echo ""
echo "===== Next Steps ====="
echo "1. Run the test connection script: python test_asyncpg_connection.py"
echo "2. If still having issues, check:"
echo "   - The username format (should be username@servername)"
echo "   - Password is correct"
echo "   - Azure PostgreSQL server allows connections (check status in Azure Portal)"
echo "   - Your IP address is allowed in the firewall rules"
echo ""