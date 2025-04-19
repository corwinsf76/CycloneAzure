#!/bin/bash

echo "=== Fixing Azure DevOps Repository URL ==="

# Show current remote
echo "Current remote configuration:"
git remote -v

# The correct repository URL from Azure DevOps
CORRECT_URL="https://justinlaughlin@dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2"
echo "Correct repository URL: $CORRECT_URL"

# Update the remote
echo "Updating remote URL..."
git remote set-url origin "$CORRECT_URL"

# Verify the update
echo "New remote configuration:"
git remote -v

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

echo "Remote URL updated successfully! You can now push with:"
echo "  git push -u origin $BRANCH"
