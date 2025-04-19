#!/bin/bash
# setup_postgresql.sh - Script to create an Azure PostgreSQL database for Cyclone v2 Trading Bot

echo "=== Setting up Azure PostgreSQL Database for Cyclone v2 Trading Bot ==="

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Azure CLI is not installed. Installing now..."
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    
    if [ $? -ne 0 ]; then
        echo "Failed to install Azure CLI. Please install it manually."
        exit 1
    fi
fi

# Login to Azure if not already logged in
echo "Checking Azure login status..."
az account show &> /dev/null
if [ $? -ne 0 ]; then
    echo "Please login to your Azure account:"
    az login
    
    if [ $? -ne 0 ]; then
        echo "Failed to login to Azure. Please try again."
        exit 1
    fi
fi

# Prompt for configuration values
read -p "Resource Group Name [RG-CycloneV2]: " RESOURCE_GROUP
RESOURCE_GROUP=${RESOURCE_GROUP:-RG-CycloneV2}

read -p "Location [eastus]: " LOCATION
LOCATION=${LOCATION:-eastus}

read -p "Server Name [cyclonev2-postgres]: " SERVER_NAME
SERVER_NAME=${SERVER_NAME:-cyclonev2-postgres}

read -p "Admin Username [cycloneadmin]: " ADMIN_USERNAME
ADMIN_USERNAME=${ADMIN_USERNAME:-cycloneadmin}

read -p "Admin Password [YourSecurePassword123!]: " ADMIN_PASSWORD
ADMIN_PASSWORD=${ADMIN_PASSWORD:-YourSecurePassword123!}

read -p "Database Name [cyclonev2db]: " DB_NAME
DB_NAME=${DB_NAME:-cyclonev2db}

# Create resource group if it doesn't exist
echo "Creating resource group $RESOURCE_GROUP in $LOCATION if it doesn't exist..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create PostgreSQL flexible server with proper SKU
# Using Burstable tier which is more cost-effective for development/testing
echo "Creating PostgreSQL flexible server $SERVER_NAME..."
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name $SERVER_NAME \
  --location $LOCATION \
  --admin-user $ADMIN_USERNAME \
  --admin-password "$ADMIN_PASSWORD" \
  --tier Burstable \
  --sku-name Standard_B1ms \
  --storage-size 32 \
  --version 13

# Create database
echo "Creating database $DB_NAME..."
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name $SERVER_NAME \
  --database-name $DB_NAME

# Configure firewall rules to allow your current IP
echo "Configuring firewall rules..."
MY_IP=$(curl -s https://api.ipify.org)
echo "Your current IP address is: $MY_IP"

az postgres flexible-server firewall-rule create \
  --resource-group $RESOURCE_GROUP \
  --name $SERVER_NAME \
  --rule-name AllowMyIP \
  --start-ip-address $MY_IP \
  --end-ip-address $MY_IP

# Allow Azure services (optional)
read -p "Allow connections from Azure services? (y/n) [y]: " ALLOW_AZURE
ALLOW_AZURE=${ALLOW_AZURE:-y}

if [[ $ALLOW_AZURE == "y" ]]; then
  az postgres flexible-server firewall-rule create \
    --resource-group $RESOURCE_GROUP \
    --name $SERVER_NAME \
    --rule-name AllowAzureServices \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0
fi

# Create a new .env file with the PostgreSQL connection string
CONNECTION_STRING="postgresql://$ADMIN_USERNAME:$ADMIN_PASSWORD@$SERVER_NAME.postgres.database.azure.com:5432/$DB_NAME"
echo "Creating .env file with PostgreSQL connection string..."

# Backup the old .env file if it exists
if [ -f .env ]; then
    cp .env .env.mssql.backup
    echo "Backed up existing .env file to .env.mssql.backup"
fi

# Copy from template and replace connection string
cp .env.postgresql .env
sed -i "s|DATABASE_URL=postgresql://.*|DATABASE_URL=$CONNECTION_STRING|g" .env

echo "========================================"
echo "Azure PostgreSQL database setup complete!"
echo "Connection string: $CONNECTION_STRING"
echo ""
echo "Your database configuration has been saved to .env"
echo "Run 'pip install psycopg2-binary asyncpg' to install the required Python packages"
echo "========================================"