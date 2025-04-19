#!/bin/bash

echo "==== Azure DevOps Repository Creation Script ===="
echo "This script will ensure your repository exists in Azure DevOps."

# Check remote status
echo "Current remote configuration:"
git remote -v

# Verify the repository exists
echo -e "\nVerifying Azure DevOps connection..."
ORG_NAME="justinlaughlin"
PROJECT_NAME="cyclonev2"
REPO_NAME="cyclonev2repos"

# Create the repository URL
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo "Repository URL: $REPO_URL"

# Check current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

echo -e "\n==== Repository Creation ===="
echo "You need to create your repository in Azure DevOps:"
echo "1. Go to: https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git"
echo "2. Click on 'New repository'"
echo "3. Enter '$REPO_NAME' as the repository name"
echo "4. Leave 'Add a README' unchecked"
echo "5. Click 'Create'"

read -p "Have you created the repository in Azure DevOps? (y/n): " REPO_CREATED
if [[ "$REPO_CREATED" != "y" && "$REPO_CREATED" != "Y" ]]; then
    echo "Please create the repository before continuing."
    echo "After creating it, run this script again."
    exit 1
fi

# Push to the repository
echo -e "\nPushing to Azure DevOps..."
git push -u origin $BRANCH

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed code to Azure DevOps!"
    echo "Your repository should now be available at:"
    echo "$REPO_URL"
else
    echo "❌ Failed to push to Azure DevOps."
    echo -e "\nTroubleshooting:"
    echo "1. Ensure you created the repository with exactly the name: $REPO_NAME"
    echo "2. Verify your PAT has 'Code (Read & Write)' permissions"
    echo "3. Try pushing manually with: git push -u origin $BRANCH --force"
fi
