#!/bin/bash

echo "==== Azure DevOps Setup Script for VS Code Remote Connection ===="
echo "This script is designed to be run while connected to cyclonev2-2 via VS Code"
echo "Current environment:"
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"

# Check for the problematic 'origin' file
if [ -f "origin" ]; then
    echo "Found a file named 'origin' that may conflict with Git remotes."
    echo "Renaming to 'origin.bak'..."
    mv origin origin.bak
    echo "✅ File renamed."
fi

# Check Git repository status
if [ -d ".git" ]; then
    echo "Existing Git repository found."
    
    # Check if the repository is working correctly
    if ! git status &>/dev/null; then
        echo "⚠️ Git repository appears to be corrupted."
        read -p "Would you like to reinitialize it? (y/n): " REINIT
        if [[ "$REINIT" == "y" || "$REINIT" == "Y" ]]; then
            echo "Reinitializing Git repository..."
            rm -rf .git
            git init
            echo "✅ Fresh Git repository initialized."
        else
            echo "Skipping repository reinitialization."
        fi
    else
        echo "✅ Git repository appears to be working correctly."
    fi
else
    echo "No Git repository found. Initializing a new one..."
    git init
    echo "✅ Git repository initialized."
fi

# Add files and commit if needed
if [ -n "$(git status --porcelain)" ]; then
    echo "Uncommitted changes detected."
    read -p "Add and commit all files? (y/n): " COMMIT_FILES
    if [[ "$COMMIT_FILES" == "y" || "$COMMIT_FILES" == "Y" ]]; then
        git add .
        git commit -m "Initial commit for Azure DevOps"
        echo "✅ Files committed."
    else
        echo "Skipping commit. You'll need to commit files before pushing."
    fi
fi

# Set up Azure DevOps connection
echo -e "\nSetting up Azure DevOps connection:"
read -p "Enter your Azure DevOps organization name: " ORG_NAME
read -p "Enter your Azure DevOps project name: " PROJECT_NAME
read -p "Enter your repository name: " REPO_NAME

# Create the connection
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo "Setting remote origin to: $REPO_URL"

# Remove existing origin if present
git remote remove origin 2>/dev/null

# Add new origin
git remote add origin "$REPO_URL"

# Verify remote
echo -e "\nVerifying remote configuration:"
git remote -v

# Get current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [ "$BRANCH" == "unknown" ]; then
    echo "No commits found. Please commit your changes before proceeding."
    exit 1
fi

echo -e "\nCurrent branch: $BRANCH"
if [ "$BRANCH" == "master" ]; then
    read -p "Would you like to rename your branch from 'master' to 'main'? (y/n): " RENAME_BRANCH
    if [[ "$RENAME_BRANCH" == "y" || "$RENAME_BRANCH" == "Y" ]]; then
        git branch -m master main
        BRANCH="main"
        echo "Branch renamed to 'main'."
    fi
fi

# Set up authentication
echo -e "\n==== Authentication Setup ===="
echo "To push to Azure DevOps, you need a Personal Access Token (PAT)."
echo "1. Go to: https://dev.azure.com/$ORG_NAME/_usersSettings/tokens"
echo "2. Create a new token with 'Code (Read & Write)' permissions"
echo "3. Save this token somewhere secure - you'll need it for pushing"

# Configure credential helper
read -p "Configure credential helper to store your PAT? (y/n): " CONFIG_CREDS
if [[ "$CONFIG_CREDS" == "y" || "$CONFIG_CREDS" == "Y" ]]; then
    git config credential.helper store
    echo "Credential helper configured. You'll be prompted for your PAT once."
fi

# Push option
read -p "Would you like to push to Azure DevOps now? (y/n): " PUSH_NOW
if [[ "$PUSH_NOW" == "y" || "$PUSH_NOW" == "Y" ]]; then
    echo "Pushing to Azure DevOps..."
    git push -u origin $BRANCH
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed to Azure DevOps!"
    else
        echo "❌ Push failed. Make sure your PAT has the correct permissions."
        echo "You can try again with: git push -u origin $BRANCH"
    fi
else
    echo "You can push later with: git push -u origin $BRANCH"
fi

echo -e "\n==== Setup Complete ===="
echo "Your repository is now connected to Azure DevOps."
echo "Repository URL: $REPO_URL"
echo "Branch: $BRANCH"
