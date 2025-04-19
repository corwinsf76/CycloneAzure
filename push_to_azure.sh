#!/bin/bash

echo "==== Push Code to Azure DevOps ===="
echo "This script will push your code to Azure DevOps after you've created the repository."

# Get remote URL
REMOTE_URL=$(git remote get-url origin)
echo "Target repository: $REMOTE_URL"

# Confirm repository creation
read -p "Have you created the 'cyclonev2repos' repository in Azure DevOps? (y/n): " REPO_CREATED

if [[ "$REPO_CREATED" != "y" && "$REPO_CREATED" != "Y" ]]; then
    echo "Please create the repository first following the instructions in azure_guide.sh"
    exit 1
fi

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

# Ask if user wants to push
read -p "Push code to Azure DevOps now? (y/n): " PUSH_NOW

if [[ "$PUSH_NOW" == "y" || "$PUSH_NOW" == "Y" ]]; then
    echo "Pushing to Azure DevOps..."
    git push -u origin $BRANCH
    
    # Check if push succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed to Azure DevOps!"
        echo "Your code is now available at: $REMOTE_URL"
    else
        echo "❌ Push failed. Trying with --force option..."
        read -p "Do you want to try force push? This will overwrite any existing content (y/n): " FORCE_PUSH
        
        if [[ "$FORCE_PUSH" == "y" || "$FORCE_PUSH" == "Y" ]]; then
            git push -u origin $BRANCH --force
            
            if [ $? -eq 0 ]; then
                echo "✅ Force push successful!"
                echo "Your code is now available at: $REMOTE_URL"
            else
                echo "❌ Force push also failed. Please check your credentials and repository settings."
            fi
        fi
    fi
else
    echo "You can push your code later with:"
    echo "  git push -u origin $BRANCH"
fi
