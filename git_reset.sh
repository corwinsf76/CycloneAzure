#!/bin/bash

echo "=== Git Repository Deep Fix ==="
echo "Current directory: $(pwd)"

# Check if there's a file named 'origin' in the repository
if [ -f "origin" ]; then
    echo "Found a file named 'origin' in your repository."
    echo "This is likely causing the 'invalid gitfile format' error."
    
    # Check the file content
    echo "File content:"
    cat origin
    
    echo "Renaming this file to 'origin_file.txt'..."
    mv origin origin_file.txt
    echo "✅ File renamed."
fi

# Check for unusual gitfile references
echo "Checking for unusual git references..."
find .git -type f -name "origin" -o -name "*origin*" 2>/dev/null

# List remotes in raw git config
echo "Raw git remote configuration:"
if [ -f ".git/config" ]; then
    grep -A 10 "\[remote" .git/config
else
    echo "No .git/config file found"
fi

echo -e "\n=== Complete Repository Reset ==="
echo "This will create a new git repository while preserving your files."
read -p "Do you want to continue? (y/n): " CONTINUE

if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Create a backup of the entire directory (excluding .git)
echo "Creating backup of your files..."
mkdir -p ../cyclonev2_backup
find . -type f -not -path "./.git/*" -not -path "./node_modules/*" -exec cp --parents {} ../cyclonev2_backup/ \;
echo "✅ Backup created at ../cyclonev2_backup"

# Remove git completely
echo "Removing existing git repository..."
rm -rf .git
rm -f .gitmodules

# Initialize a fresh repository
echo "Initializing a fresh git repository..."
git init
git config --global init.defaultBranch main

# Add all files
echo "Adding all files to new repository..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit after repository reset"

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

echo -e "\n✅ Git repository has been completely reset and configured for Azure DevOps."
echo "To push your code, run:"
echo "  git push -u origin main --force"
