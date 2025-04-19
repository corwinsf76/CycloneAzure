#!/bin/bash

echo "=== Git Configuration Fix Script ==="
echo "Current directory: $(pwd)"

# Check for the problematic 'origin' file
if [ -f "origin" ]; then
    echo "Found problematic 'origin' file. This is conflicting with Git remotes."
    read -p "Do you want to rename this file? (y/n): " RENAME_FILE
    if [[ "$RENAME_FILE" == "y" || "$RENAME_FILE" == "Y" ]]; then
        mv origin origin.bak
        echo "✅ File renamed to 'origin.bak'"
    else
        echo "⚠️ Skipping file rename. This may continue to cause issues."
    fi
fi

# Clean up Git configuration
echo "Cleaning up Git configuration..."
rm -f .git/FETCH_HEAD 2>/dev/null
rm -f .git/config.lock 2>/dev/null

# Reset Git remotes
echo "Resetting Git remotes..."
git remote -v
for remote in $(git remote); do
    git remote remove $remote
done

# Set up Azure DevOps connection
echo -e "\nSetting up Azure DevOps connection:"
read -p "Organization name: " ORG_NAME
read -p "Project name: " PROJECT_NAME
read -p "Repository name: " REPO_NAME

# Create the connection with HTTPS URL
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo "Adding Azure DevOps remote: $REPO_URL"
git remote add origin "$REPO_URL"

# Verify the connection
echo -e "\nVerifying connection:"
git remote -v

echo -e "\n✅ Git configuration has been fixed and Azure DevOps connection established."
echo "To push your code, run:"
echo "  git push -u origin master"
