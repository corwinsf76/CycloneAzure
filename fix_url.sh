#!/bin/bash

echo "=== Fixing Azure DevOps Repository URL ==="

# Show current remote
echo "Current remote configuration:"
git remote -v

# Remove the current remote
echo "Removing current remote..."
git remote remove origin

# Add the correct remote
echo "Adding correct remote URL..."
git remote add origin https://justinlaughlin@dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2

# Verify the update
echo "New remote configuration:"
git remote -v

echo "Remote URL has been updated. Now push with:"
echo "  git push -u origin main"
