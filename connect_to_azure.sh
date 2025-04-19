#!/bin/bash

# Azure DevOps Repository Connection Script
# This script helps connect your local repository to Azure DevOps

# Check if the file is being executed in a Git repository
if [ ! -d ".git" ]; then
    echo "Error: Not a git repository. Please run setup_new_repo.sh first or initialize git manually."
    echo "You can run: git init"
    exit 1
fi

# Configuration variables - edit these!
AZURE_ORGANIZATION=""
AZURE_PROJECT=""
REPO_NAME=""

# Ask for parameters if not hardcoded
if [ -z "$AZURE_ORGANIZATION" ]; then
    read -p "Enter your Azure DevOps organization name: " AZURE_ORGANIZATION
fi

if [ -z "$AZURE_PROJECT" ]; then
    read -p "Enter your Azure DevOps project name: " AZURE_PROJECT
fi

if [ -z "$REPO_NAME" ]; then
    read -p "Enter your repository name: " REPO_NAME
fi

echo "Connecting to Azure DevOps repository..."

# Set Git remote
REPO_URL="https://dev.azure.com/$AZURE_ORGANIZATION/$AZURE_PROJECT/_git/$REPO_NAME"
echo "Setting remote origin to: $REPO_URL"
git remote remove origin &> /dev/null
git remote add origin "$REPO_URL"

# Verify remote was added
echo "Verifying remote connection..."
git remote -v

# Configure Git credentials (optional)
read -p "Do you want to configure Git credentials for Azure DevOps? (y/n): " CONFIGURE_CREDS
if [[ "$CONFIGURE_CREDS" == "y" || "$CONFIGURE_CREDS" == "Y" ]]; then
    echo "You'll need a Personal Access Token (PAT) from Azure DevOps."
    echo "Generate one at: https://dev.azure.com/$AZURE_ORGANIZATION/_usersSettings/tokens"
    read -p "Enter your Personal Access Token: " PAT
    
    # Store credentials
    git config credential.helper store
    echo "Credentials configured."
fi

# Push code to Azure repository
echo "Ready to push your code to Azure DevOps."
echo "Run the following commands:"
echo "  git branch -M main"
echo "  git push -u origin main"

read -p "Push code now? (y/n): " PUSH_NOW
if [[ "$PUSH_NOW" == "y" || "$PUSH_NOW" == "Y" ]]; then
    git branch -M main
    git push -u origin main
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed to Azure DevOps repository!"
    else
        echo "❌ Failed to push to repository. Check your credentials and try again."
        echo "You can push manually using: git push -u origin main"
    fi
else
    echo "Skipping push. You can push manually later using:"
    echo "  git push -u origin main"
fi

echo "Repository connection to Azure DevOps completed!"
