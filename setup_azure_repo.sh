#!/bin/bash

echo "=== Complete Azure Repository Setup ==="
echo "This script will help you set up your repository for Azure DevOps step-by-step."

# Step 1: Check if we're in a Git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a Git repository root directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Step 2: Commit any pending changes
echo -e "\n=== Step 1: Checking for uncommitted changes ==="
if [ -n "$(git status --porcelain)" ]; then
    echo "Uncommitted changes detected."
    read -p "Do you want to commit these changes? (y/n): " COMMIT_CHANGES
    if [[ "$COMMIT_CHANGES" == "y" || "$COMMIT_CHANGES" == "Y" ]]; then
        git add .
        git commit -m "Commit pending changes before Azure setup"
        echo "✅ Changes committed successfully."
    else
        echo "⚠️ Proceeding with uncommitted changes. This might cause issues."
    fi
else
    echo "✅ No uncommitted changes detected."
fi

# Step 3: Check and fix cyclonev22 directory issue
echo -e "\n=== Step 2: Checking for embedded repository issues ==="
if [ -d "cyclonev22" ] && [ -d "cyclonev22/.git" ]; then
    echo "Found embedded Git repository at cyclonev22/"
    read -p "Do you want to: (1) Remove it, (2) Convert to a regular directory, or (3) Skip? [1/2/3]: " REPO_CHOICE
    
    case $REPO_CHOICE in
        1)
            echo "Removing cyclonev22 directory entirely..."
            git rm --cached cyclonev22
            rm -rf cyclonev22
            git commit -m "Remove cyclonev22 embedded repository"
            echo "✅ cyclonev22 directory removed."
            ;;
        2)
            echo "Converting cyclonev22 to a regular directory..."
            git rm --cached cyclonev22
            rm -rf cyclonev22/.git
            git add cyclonev22
            git commit -m "Convert cyclonev22 to regular directory"
            echo "✅ cyclonev22 converted to a regular directory."
            ;;
        3)
            echo "Skipping cyclonev22 handling."
            ;;
    esac
else
    echo "✅ No embedded Git repository found at cyclonev22/ or directory doesn't exist."
fi

# Step 4: Rename branch if needed
echo -e "\n=== Step 3: Checking branch name ==="
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"
if [ "$CURRENT_BRANCH" = "master" ]; then
    read -p "Do you want to rename the branch from 'master' to 'main'? (y/n): " RENAME_BRANCH
    if [[ "$RENAME_BRANCH" == "y" || "$RENAME_BRANCH" == "Y" ]]; then
        git branch -m master main
        echo "✅ Branch renamed to 'main'."
        CURRENT_BRANCH="main"
    else
        echo "Keeping branch name as 'master'."
    fi
fi

# Step 5: Connect to Azure DevOps
echo -e "\n=== Step 4: Setting up Azure DevOps connection ==="
read -p "Do you want to connect to Azure DevOps now? (y/n): " CONNECT_AZURE
if [[ "$CONNECT_AZURE" == "y" || "$CONNECT_AZURE" == "Y" ]]; then
    read -p "Enter your Azure DevOps organization name: " AZURE_ORGANIZATION
    read -p "Enter your Azure DevOps project name: " AZURE_PROJECT
    read -p "Enter your repository name: " REPO_NAME
    
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
        echo "✅ Credentials configured."
    fi
    
    # Push code
    read -p "Push code to Azure DevOps now? (y/n): " PUSH_NOW
    if [[ "$PUSH_NOW" == "y" || "$PUSH_NOW" == "Y" ]]; then
        git push -u origin $CURRENT_BRANCH
        if [ $? -eq 0 ]; then
            echo "✅ Successfully pushed to Azure DevOps repository!"
        else
            echo "❌ Failed to push to repository. Check your credentials and try again."
            echo "   You can push manually using: git push -u origin $CURRENT_BRANCH"
        fi
    else
        echo "Skipping push. You can push manually later using:"
        echo "  git push -u origin $CURRENT_BRANCH"
    fi
else
    echo "Skipping Azure DevOps connection. You can set this up later."
fi

echo -e "\n=== Repository Setup Completed ✅ ==="
echo "Your Git repository has been prepared and configured for Azure DevOps."
