#!/bin/bash

echo "==== Azure DevOps Repository Creation Guide ===="
echo "Your Git remote is correctly configured to:"
echo "  https://dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2repos"
echo ""
echo "However, you need to create this repository in Azure DevOps first."

# Extract information from the remote URL
REMOTE_URL=$(git remote get-url origin)
ORG_NAME=$(echo $REMOTE_URL | cut -d'/' -f4)
PROJECT_NAME=$(echo $REMOTE_URL | cut -d'/' -f5)
REPO_NAME=$(echo $REMOTE_URL | cut -d'/' -f7)

echo "Organization: $ORG_NAME"
echo "Project: $PROJECT_NAME"
echo "Repository: $REPO_NAME"

echo -e "\n==== Steps to Create Repository in Azure DevOps ===="
echo "1. Go to: https://dev.azure.com/$ORG_NAME"
echo "2. Click on your project: $PROJECT_NAME"
echo "   (If the project doesn't exist, create it first)"
echo "3. Go to Repos > Files"
echo "4. At the top, you'll see a dropdown with repositories"
echo "5. Click on the dropdown and select 'New repository'"
echo "6. Enter '$REPO_NAME' as the name"
echo "7. Do NOT initialize with a README or .gitignore"
echo "8. Click 'Create'"

read -p "Have you created the repository in Azure DevOps? (y/n): " REPO_CREATED

if [[ "$REPO_CREATED" == "y" || "$REPO_CREATED" == "Y" ]]; then
    echo -e "\n==== Pushing Your Code to Azure DevOps ===="
    echo "Pushing to Azure DevOps..."
    
    # Get current branch
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    
    # Push the code
    git push -u origin $BRANCH
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed your code to Azure DevOps!"
        echo "Your repository is now available at:"
        echo "  https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
    else
        echo "❌ Failed to push your code."
        echo "Common issues:"
        echo "  - Repository name might be incorrect"
        echo "  - You might need to use --force if the repository was initialized"
        echo "  - PAT might not have correct permissions"
        echo ""
        echo "Try pushing manually with:"
        echo "  git push -u origin $BRANCH --force"
    fi
else
    echo "Please create the repository first, then run this script again."
fi
