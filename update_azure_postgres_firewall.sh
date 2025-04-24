#!/bin/bash
# Script to update Azure PostgreSQL server firewall rules to allow your current IP address

echo "===== Azure PostgreSQL Firewall Updater ====="
echo "This script will add your current IP address to the firewall rules of your Azure PostgreSQL server."

# PostgreSQL server details
SERVER_NAME="cyclonev2"
FULL_SERVER_NAME="cyclonev2.postgres.database.azure.com"

# Get the current IP address
echo "Detecting your current public IP address..."
CURRENT_IP=$(curl -s https://api.ipify.org)

if [ -z "$CURRENT_IP" ]; then
    echo "✗ Failed to detect your IP address"
    echo "Please enter your IP address manually:"
    read CURRENT_IP
else
    echo "✓ Your current public IP address: $CURRENT_IP"
fi

# Check if we have the Azure CLI
if ! command -v az &> /dev/null; then
    echo "✗ Azure CLI (az) is not installed or not in your PATH"
    echo "  To install the Azure CLI, see: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    echo "  You'll need to add the firewall rule manually in the Azure Portal:"
    echo "  1. Go to Azure Portal: https://portal.azure.com"
    echo "  2. Find your PostgreSQL server: $FULL_SERVER_NAME"
    echo "  3. Go to 'Security' → 'Networking'"
    echo "  4. Add a new firewall rule with your IP: $CURRENT_IP"
    exit 1
fi

# Check if logged in to Azure
echo "Checking Azure CLI login status..."
az account show &> /dev/null
if [ $? -ne 0 ]; then
    echo "✗ You are not logged in to Azure CLI"
    echo "  Please run: az login"
    echo "  Then run this script again"
    exit 1
fi

echo "✓ You are logged in to Azure CLI"

# Try to find the resource group
echo "Searching for PostgreSQL server in your subscriptions..."
RESOURCE_GROUP=$(az postgres server list --query "[?name=='$SERVER_NAME'].resourceGroup" -o tsv)

if [ -z "$RESOURCE_GROUP" ]; then
    echo "✗ Could not automatically find the resource group for $SERVER_NAME"
    echo "  Please enter the resource group name manually:"
    read RESOURCE_GROUP
else
    echo "✓ Found resource group: $RESOURCE_GROUP"
fi

# Create a unique rule name based on date and IP
RULE_NAME="Allow_${CURRENT_IP//./_}_$(date +%Y%m%d)"

# Add the firewall rule
echo "Adding firewall rule to allow your IP address ($CURRENT_IP)..."
az postgres server firewall-rule create \
    --resource-group "$RESOURCE_GROUP" \
    --server-name "$SERVER_NAME" \
    --name "$RULE_NAME" \
    --start-ip-address "$CURRENT_IP" \
    --end-ip-address "$CURRENT_IP"

if [ $? -eq 0 ]; then
    echo "✓ Successfully added firewall rule to allow your IP address"
    echo "  You should now be able to connect to your database"
else
    echo "✗ Failed to add firewall rule automatically"
    echo "  You'll need to add the firewall rule manually in the Azure Portal:"
    echo "  1. Go to Azure Portal: https://portal.azure.com"
    echo "  2. Find your PostgreSQL server: $FULL_SERVER_NAME"
    echo "  3. Go to 'Security' → 'Networking'"
    echo "  4. Add a new firewall rule with your IP: $CURRENT_IP"
fi

echo ""
echo "===== Next Steps ====="
echo "1. Run the connection test script: python test_postgres_connection.py"
echo "2. If the connection works, try running the dashboard: python dashboard/launch_dashboard.py"
echo ""