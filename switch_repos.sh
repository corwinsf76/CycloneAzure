#!/bin/bash

echo "=== Repository Connection Switcher ==="
echo "This script will help disconnect from the old repository and connect to Azure DevOps."

# Check current remotes
echo "Current repository remotes:"
git remote -v

# Remove existing remotes
echo -e "\nRemoving existing remote connections..."
for remote in $(git remote); do
    echo "Removing remote: $remote"
    git remote remove $remote
done

# Set up Azure DevOps remote
echo -e "\nSetting up Azure DevOps connection:"
read -p "Enter your Azure DevOps organization name: " AZURE_ORGANIZATION
read -p "Enter your Azure DevOps project name: " AZURE_PROJECT
read -p "Enter your repository name: " REPO_NAME

# Configure new remote
REPO_URL="https://dev.azure.com/$AZURE_ORGANIZATION/$AZURE_PROJECT/_git/$REPO_NAME"
echo "Adding Azure DevOps remote: $REPO_URL"
git remote add origin "$REPO_URL"

# Verify new remote
echo -e "\nVerifying new remote connection:"
git remote -v

echo -e "\n=== Repository Connection Updated ==="
echo "Your repository is now connected to Azure DevOps."
echo "To push your code, use: git push -u origin main"
