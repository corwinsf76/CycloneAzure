#!/bin/bash

echo "==== Local Azure DevOps Repository Setup ===="
echo "This script will fix your local repository and connect to Azure DevOps."
echo "Current directory: $(pwd)"

# Check if we have a file named 'origin' that's causing problems
if [ -f "origin" ]; then
    echo "Found file named 'origin' which conflicts with Git remotes."
    echo "Renaming to 'origin.bak'..."
    mv origin origin.bak
    echo "✅ File renamed."
fi

# Option to reinitialize git repository
echo -e "\nWould you like to reinitialize your Git repository?"
echo "This will preserve all your files but create a fresh Git history."
read -p "Reinitialize Git? (y/n): " REINIT_GIT

if [[ "$REINIT_GIT" == "y" || "$REINIT_GIT" == "Y" ]]; then
    echo "Backing up any important Git hooks..."
    mkdir -p .git_backup
    if [ -d ".git/hooks" ]; then
        cp -r .git/hooks .git_backup/hooks
    fi
    
    echo "Removing existing Git repository..."
    rm -rf .git
    
    echo "Initializing fresh Git repository..."
    git init
    
    echo "Adding all files..."
    git add .
    
    echo "Creating initial commit..."
    git commit -m "Initial commit"
    
    echo "✅ Git repository reinitialized."
fi

# Set up connection to Azure DevOps
echo -e "\nSetting up connection to Azure DevOps:"
read -p "Enter your Azure DevOps organization name: " ORG_NAME
read -p "Enter your Azure DevOps project name: " PROJECT_NAME
read -p "Enter your repository name: " REPO_NAME

# Create Azure DevOps URL
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo "Setting remote origin to: $REPO_URL"

# Remove any existing remote called 'origin'
git remote remove origin 2>/dev/null

# Add the new remote
git remote add origin "$REPO_URL"

# Verify the remote was added
echo -e "\nVerifying remote connection:"
git remote -v

# Set up credential helper
echo -e "\nConfiguring credentials..."
git config credential.helper store
echo "Credential helper configured."

# Get current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "\nCurrent branch: $BRANCH"

# Ask if user wants to rename branch to main
if [ "$BRANCH" == "master" ]; then
    read -p "Would you like to rename your branch from 'master' to 'main'? (y/n): " RENAME_BRANCH
    if [[ "$RENAME_BRANCH" == "y" || "$RENAME_BRANCH" == "Y" ]]; then
        git branch -m master main
        BRANCH="main"
        echo "Branch renamed to 'main'."
    fi
fi

# Inform about authentication
echo -e "\n==== Authentication Information ===="
echo "To push to Azure DevOps, you need a Personal Access Token (PAT)."
echo "Create one at: https://dev.azure.com/$ORG_NAME/_usersSettings/tokens"
echo "Give it 'Code (Read & Write)' permissions at minimum."

read -p "Do you have a PAT ready? (y/n): " HAVE_PAT
if [[ "$HAVE_PAT" == "y" || "$HAVE_PAT" == "Y" ]]; then
    # Ask if user wants to push now
    read -p "Push to Azure DevOps now? (y/n): " PUSH_NOW
    if [[ "$PUSH_NOW" == "y" || "$PUSH_NOW" == "Y" ]]; then
        echo "Pushing to Azure DevOps..."
        git push -u origin $BRANCH
        if [ $? -eq 0 ]; then
            echo "✅ Successfully pushed to Azure DevOps!"
        else
            echo "❌ Failed to push to Azure DevOps."
            echo "You may need to try again manually with: git push -u origin $BRANCH"
        fi
    else
        echo "You can push later with: git push -u origin $BRANCH"
    fi
else
    echo "Please create a PAT before pushing to Azure DevOps."
    echo "Once you have your PAT, push with: git push -u origin $BRANCH"
fi

echo -e "\n==== Setup Complete ===="
echo "Your repository is now set up for Azure DevOps."
