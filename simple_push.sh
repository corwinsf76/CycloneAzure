#!/bin/bash

# This is a minimal script to push to Azure DevOps
echo "=== Simple Azure DevOps Push ==="

# Show current Git status
echo "Current Git configuration:"
git remote -v
echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"

# Directly attempt to push
echo "Attempting push to Azure DevOps..."
git push -u origin $(git rev-parse --abbrev-ref HEAD)

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Push successful!"
else
    echo "❌ Push failed."
    echo
    echo "Try these manual commands:"
    echo "1. Regular push:"
    echo "   git push -u origin $(git rev-parse --abbrev-ref HEAD)"
    echo
    echo "2. Force push (if regular push fails):"
    echo "   git push -u origin $(git rev-parse --abbrev-ref HEAD) --force"
    echo
    echo "Make sure you've created the repository at:"
    echo "https://dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2repos"
fi
