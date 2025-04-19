#!/bin/bash

# Simple script to connect to Azure DevOps - minimal version

echo "======================================"
echo "Simple Azure DevOps Connection Script"
echo "======================================"

# Check Git status 
echo "Current Git status:"
git status
echo ""

# Get Azure DevOps details
read -p "Enter your Azure DevOps organization name: " AZURE_ORGANIZATION
read -p "Enter your Azure DevOps project name: " AZURE_PROJECT
read -p "Enter your repository name: " REPO_NAME

# Set Git remote
REPO_URL="https://dev.azure.com/$AZURE_ORGANIZATION/$AZURE_PROJECT/_git/$REPO_NAME"
echo "Setting remote origin to: $REPO_URL"
git remote remove origin 2>/dev/null
git remote add origin "$REPO_URL"

# Verify remote was added
echo "Verifying remote connection:"
git remote -v

echo -e "\nRemote 'origin' has been set to your Azure DevOps repository"
echo "To push your code, run: git push -u origin master"
echo "======================================"
