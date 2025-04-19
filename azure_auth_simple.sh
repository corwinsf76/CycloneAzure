#!/bin/bash

echo "=== Azure DevOps Simple Authentication Setup ==="

# Configure credential helper
git config --global credential.helper store
echo "Credential helper configured to store credentials."

echo -e "\nWhen you push to Azure DevOps, you'll be prompted for:"
echo "- Username: Your Azure DevOps email"
echo "- Password: Your Personal Access Token (PAT)"
echo -e "\nAfter entering these once, git will remember them."
echo -e "\nTo push your code with stored credentials:"
echo "  git push -u origin master"
