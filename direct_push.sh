#!/bin/bash

echo "=== Direct Azure DevOps Push Script ==="
echo "Current directory: $(pwd)"
echo "Current Git configuration:"
git remote -v
echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"

echo -e "\nThis script will push your code directly to Azure DevOps using the existing configuration."
echo "Make sure you've already created the repository in Azure DevOps at:"
echo "  https://dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2repos"

read -p "Ready to push? (y/n): " READY
if [[ "$READY" != "y" && "$READY" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo -e "\nPushing to Azure DevOps..."
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Try regular push first
git push -u origin $BRANCH

if [ $? -ne 0 ]; then
    echo -e "\nRegular push failed. Let's try force push..."
    read -p "Use force push? (y/n): " FORCE
    if [[ "$FORCE" == "y" || "$FORCE" == "Y" ]]; then
        git push -u origin $BRANCH --force
        if [ $? -eq 0 ]; then
            echo "Force push successful!"
        else
            echo "Force push also failed."
        fi
    fi
fi

echo -e "\nIf push failed, here are some troubleshooting steps:"
echo "1. Make sure the repository exists in Azure DevOps"
echo "2. Check your credentials:"
echo "   - Username: Your email associated with Azure DevOps"
echo "   - Password: Your Personal Access Token (not your regular password)"
echo "3. Create a PAT at: https://dev.azure.com/justinlaughlin/_usersSettings/tokens"
echo "4. Try pushing manually with: git push -u origin $BRANCH --force"
