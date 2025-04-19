#!/bin/bash

echo "=== Creating a Fresh Repository ==="
echo "This script will create a completely new repository in a temporary directory."

# Create a new directory
TEMP_DIR="../cyclonev2_fresh"
echo "Creating a fresh directory at $TEMP_DIR"
mkdir -p $TEMP_DIR

# Copy all files (excluding git-related files)
echo "Copying your files to the fresh directory..."
rsync -av --exclude='.git/' --exclude='.gitmodules' --exclude='node_modules/' --exclude='origin' . $TEMP_DIR/

# Initialize git in the fresh directory
echo "Initializing git in the fresh directory..."
cd $TEMP_DIR
git init
git config --global init.defaultBranch main

# Add and commit files
echo "Adding and committing files..."
git add .
git commit -m "Initial commit in fresh repository"

# Set up Azure DevOps connection
echo -e "\nSetting up Azure DevOps connection:"
read -p "Organization name: " ORG_NAME
read -p "Project name: " PROJECT_NAME
read -p "Repository name: " REPO_NAME

# Create the connection
REPO_URL="https://dev.azure.com/$ORG_NAME/$PROJECT_NAME/_git/$REPO_NAME"
echo "Adding Azure DevOps remote: $REPO_URL"
git remote add origin "$REPO_URL"

# Verify remote
echo "Verifying remote configuration:"
git remote -v

echo -e "\n✅ Fresh repository created and configured at $TEMP_DIR"
echo "Your original files remain untouched in $(pwd)/../cyclonev2"
echo 
echo "Next steps:"
echo "1. cd $TEMP_DIR"
echo "2. git push -u origin main"
echo "3. Once confirmed working, you can replace your original directory"
