#!/bin/bash

# Azure DevOps Git Reference Guide
echo "=== Azure DevOps Git Reference ==="
echo "Your repository is now connected to: https://dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2"

echo -e "\n== Common Git Commands for Azure DevOps =="
echo "# Pull latest changes"
echo "git pull origin main"
echo ""
echo "# Create a new branch"
echo "git checkout -b feature/your-feature-name"
echo ""
echo "# Push changes"
echo "git add ."
echo "git commit -m \"Your commit message\""
echo "git push origin main"
echo ""
echo "# Create and push a tag (for releases)"
echo "git tag v1.0.0"
echo "git push origin v1.0.0"

echo -e "\n== Azure DevOps Integration =="
echo "# Configure CI/CD Pipeline: Go to Pipelines > New Pipeline in Azure DevOps"
echo "# Create Work Items: Go to Boards > Work Items in Azure DevOps"
echo "# View Repository: Go to Repos > Files in Azure DevOps"

echo -e "\n== Credential Management =="
echo "# Your credentials are now stored and should work automatically"
echo "# If you need to update your PAT, create a new one in Azure DevOps and use it"
echo "# when git prompts for credentials again"
