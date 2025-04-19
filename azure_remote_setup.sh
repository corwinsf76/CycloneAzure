#!/bin/bash

echo "==== Azure Remote Repository Setup ===="
echo "This script will help you set up a connection to Azure DevOps from an SSH environment."

# Check if we're on SSH or local
echo "Current environment:"
echo "Hostname: $(hostname)"
echo "Username: $(whoami)"
echo "Working directory: $(pwd)"

echo -e "\nPlease confirm the following:"
echo "1. You're currently SSH'ed into the machine where your code lives (cyclonev2-2)"
echo "2. You're in the directory containing your project files"

read -p "Are you in the correct environment? (y/n): " CORRECT_ENV
if [[ "$CORRECT_ENV" != "y" && "$CORRECT_ENV" != "Y" ]]; then
    echo "Please SSH into cyclonev2-2 first with:"
    echo "  ssh username@cyclonev2-2"
    echo "Then navigate to your project directory and run this script again."
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git not found. Please install git first:"
    echo "  sudo apt-get update && sudo apt-get install git"
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "No git repository found in current directory."
    read -p "Initialize a new git repository? (y/n): " INIT_GIT
    if [[ "$INIT_GIT" == "y" || "$INIT_GIT" == "Y" ]]; then
        git init
        echo "Git repository initialized."
    else
        echo "Cannot proceed without a git repository. Exiting."
        exit 1
    fi
fi

# Check for problematic 'origin' file
if [ -f "origin" ]; then
    echo "Found a file named 'origin' that may conflict with git remotes."
    read -p "Rename this file to 'origin.bak'? (y/n): " RENAME_ORIGIN
    if [[ "$RENAME_ORIGIN" == "y" || "$RENAME_ORIGIN" == "Y" ]]; then
        mv origin origin.bak
        echo "File renamed to 'origin.bak'."
    fi
fi

# Check current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "none")
echo "Current branch: $CURRENT_BRANCH"

# If no commits exist, create initial commit
if [ "$CURRENT_BRANCH" == "none" ] || ! git rev-parse HEAD &> /dev/null; then
    echo "No commits found. Creating initial commit..."
    git add .
    git commit -m "Initial commit"
    echo "Initial commit created."
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "Current branch: $CURRENT_BRANCH"
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
echo "Verifying remote configuration:"
git remote -v

# Configure Git credentials
echo -e "\nSetting up Git credentials:"
echo "You'll need a Personal Access Token (PAT) from Azure DevOps."
echo "Get one at: https://dev.azure.com/$ORG_NAME/_usersSettings/tokens"
read -p "Have you already created a PAT? (y/n): " HAVE_PAT

if [[ "$HAVE_PAT" != "y" && "$HAVE_PAT" != "Y" ]]; then
    echo "Please create a PAT with 'Code (Read & Write)' permissions and run the final step manually."
else
    echo "Git will prompt for credentials when you push. Use your email and PAT as password."
    git config credential.helper store
    echo "Credential helper configured."
fi

echo -e "\n==== Setup Complete ====="
echo "To push your code to Azure DevOps, run:"
if [ "$CURRENT_BRANCH" == "master" ]; then
    echo "You're on 'master' branch. Consider renaming to 'main':"
    echo "  git branch -m master main"
    echo "  git push -u origin main"
else
    echo "  git push -u origin $CURRENT_BRANCH"
fi
echo "Note: When prompted for password, use your Personal Access Token, not your regular password."
