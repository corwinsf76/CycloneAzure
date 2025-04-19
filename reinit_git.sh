#!/bin/bash

echo "=== Reinitialize Git Repository ==="
echo "WARNING: This will reset your Git configuration but preserve your files."
read -p "Continue? (y/n): " CONTINUE

if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Backup any important Git config
if [ -d ".git" ]; then
    mkdir -p .git_backup
    cp -r .git/hooks .git_backup/ 2>/dev/null
    echo "Git hooks backed up to .git_backup/"
fi

# Remove existing Git directory
echo "Removing existing Git configuration..."
rm -rf .git

# Initialize new repository
echo "Initializing new Git repository..."
git init
git config --global init.defaultBranch main

# Add all files
echo "Adding all files to repository..."
git add .

# Initial commit
echo "Creating initial commit..."
git commit -m "Initial commit after repository reinitialization"

# Set up Azure DevOps connection
echo -e "\nSetting up Azure DevOps connection:"
read -p "Organization name: " ORG_NAME
read -p "Project name: " PROJECT_NAME
read -p "Repository name: " REPO_NAME

# Create the connection
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo "Adding Azure DevOps remote: $REPO_URL"
git remote add origin "$REPO_URL"

echo -e "\n✅ Git repository reinitialized and connected to Azure DevOps."
echo "To push your code, run:"
echo "  git push -u origin main --force"
