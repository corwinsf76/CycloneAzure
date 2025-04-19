#!/bin/bash

echo "=========================================================="
echo "      LOCAL AZURE DEVOPS REPOSITORY SETUP SCRIPT"
echo "=========================================================="
echo "This script should be run on your LOCAL MACHINE"
echo "in the directory /home/Justin/projects/cyclonev2"
echo "Current location: $(pwd)"
echo "Current user: $(whoami)"
echo "=========================================================="

# Check if we're in the right directory
if [[ "$(pwd)" != *"cyclonev2"* ]]; then
    echo "⚠️ WARNING: You may not be in the cyclonev2 directory!"
    echo "Current directory: $(pwd)"
    read -p "Continue anyway? (y/n): " CONTINUE
    if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
        echo "Please change to your project directory first."
        exit 1
    fi
fi

echo "Step 1: Checking for the problematic 'origin' file..."
if [ -f "origin" ]; then
    echo "Found problematic 'origin' file. This will be renamed."
    mv origin origin.bak
    echo "✅ Renamed 'origin' to 'origin.bak'"
fi

echo -e "\nStep 2: Let's check your Git repository status..."
if [ -d ".git" ]; then
    echo "✅ Git repository exists"
    
    # Check for any issues
    git status &>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️ Your Git repository appears to be corrupted."
        read -p "Would you like to reinitialize it? (y/n): " REINIT
        if [[ "$REINIT" == "y" || "$REINIT" == "Y" ]]; then
            rm -rf .git
            git init
            git add .
            git commit -m "Initial commit after repository reset"
            echo "✅ Repository reinitialized"
        else
            echo "⚠️ Continuing with potentially corrupted repository..."
        fi
    fi
else
    echo "No Git repository found. Creating one..."
    git init
    git add .
    git commit -m "Initial commit"
    echo "✅ Git repository created"
fi

echo -e "\nStep 3: Setting up Azure DevOps connection..."
read -p "Enter your Azure DevOps organization name: " ORG_NAME
read -p "Enter your Azure DevOps project name: " PROJECT_NAME
read -p "Enter your repository name: " REPO_NAME

# Set up the connection
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo "Setting remote origin to: $REPO_URL"
git remote remove origin &>/dev/null
git remote add origin "$REPO_URL"

# Verify connection
echo -e "\nVerifying connection:"
git remote -v

# Get branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "\nCurrent branch: $BRANCH"

# Inform about authentication
echo -e "\n=========================================================="
echo "AUTHENTICATION INFORMATION"
echo "=========================================================="
echo "To push to Azure DevOps, you'll need a Personal Access Token (PAT)."
echo "1. Go to: https://dev.azure.com/$ORG_NAME/_usersSettings/tokens"
echo "2. Create a new token with 'Code (Read & Write)' permissions"
echo "3. Copy the token - you'll need it when pushing"
echo "4. When prompted for password, use the token instead of your password"
echo "=========================================================="

# Configure credential helper
git config credential.helper store
echo "Credential helper configured to store your credentials."

echo -e "\nSetup complete! To push your code to Azure DevOps, run:"
echo "  git push -u origin $BRANCH"
echo
echo "If you prefer to use 'main' as your branch name instead of '$BRANCH', run:"
echo "  git branch -m $BRANCH main"
echo "  git push -u origin main"
