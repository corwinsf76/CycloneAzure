#!/bin/bash

echo "==== Azure DevOps Repository Creation Guide ===="
echo "Your local repository is configured correctly but you need to create the repository in Azure DevOps."

# Display configured remote
REMOTE_URL=$(git remote get-url origin)
echo "Configured repository URL: $REMOTE_URL"

echo -e "\nFollow these steps in your web browser:"
echo "------------------------------------------------------"
echo "1. Visit: https://dev.azure.com/justinlaughlin"
echo "2. Select your 'cyclonev2' project (or create it if needed)"
echo "3. Click on 'Repos' in the left sidebar"
echo "4. Click on the repository dropdown at the top of the page"
echo "5. Select 'New repository'"
echo "6. Enter 'cyclonev2repos' as the name"
echo "7. Leave 'Add a README' and other options unchecked"
echo "8. Click 'Create'"
echo "------------------------------------------------------"

echo -e "\nAfter creating the repository, run this command to push your code:"
echo "  git push -u origin main"

echo -e "\nIf you get an error about unrelated histories, use:"
echo "  git push -u origin main --force"

echo -e "\nYour code will then be available at:"
echo "  https://dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2repos"
