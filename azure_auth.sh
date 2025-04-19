#!/bin/bash

echo "=== Azure DevOps Authentication Setup ==="

echo "This script will help you set up authentication for Azure DevOps."
echo "You'll need to create a Personal Access Token (PAT) in Azure DevOps."
echo "Instructions:"
echo "1. Go to https://dev.azure.com/{your-organization}/_usersSettings/tokens"
echo "2. Click 'New Token'"
echo "3. Name it something like 'CycloneV2'"
echo "4. Select 'Full access' or at minimum 'Code (Read & Write)'"
echo "5. Click 'Create' and copy the token"
echo ""

read -p "Have you created a PAT? (y/n): " CREATED_PAT

if [[ "$CREATED_PAT" != "y" && "$CREATED_PAT" != "Y" ]]; then
    echo "Please create a PAT first and run this script again."
    exit 1
fi

read -p "Enter your Azure DevOps username/email: " USERNAME
read -sp "Enter your Personal Access Token: " PAT
echo ""

# Configure Git credentials
git config credential.helper store
echo "https://$USERNAME:$PAT@dev.azure.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

echo "✅ Credentials configured successfully."
echo "You should now be able to push to Azure DevOps without entering credentials each time."
