#!/bin/bash

echo "=== Connect to Azure DevOps Repository ==="
echo "Current directory: $(pwd)"

# Check if we're in a Git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a Git repository root directory."
    git init
    echo "✅ Git repository initialized."
    git add .
    git commit -m "Initial commit"
    echo "✅ Initial commit created."
fi

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
echo "Current branch: $CURRENT_BRANCH"

# Azure DevOps connection
echo -e "\nEnter Azure DevOps details:"
read -p "Organization name: " ORG_NAME
read -p "Project name: " PROJECT_NAME
read -p "Repository name: " REPO_NAME

# Create the connection
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo -e "\nConnecting to: $REPO_URL"
git remote add origin "$REPO_URL"

# Verify the connection
echo -e "\nVerifying connection:"
git remote -v

echo -e "\n✅ Connection to Azure DevOps established."
echo "To push your code, run:"
echo "  git push -u origin $CURRENT_BRANCH"
