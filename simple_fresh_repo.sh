#!/bin/bash

echo "==== Simple Fresh Repository Script ===="
echo "Current directory: $(pwd)"

# Create a new directory without using variables
echo "Creating a fresh directory..."
mkdir -p ../cyclonev2_simple_fresh
echo "Directory created."

# Copy files using simple copy commands
echo "Copying files (this may take a moment)..."
cp -r ./*.py ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No .py files found"
cp -r ./*.md ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No .md files found"
cp -r ./*.sh ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No .sh files found"
cp -r ./*.txt ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No .txt files found"
cp -r ./*.json ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No .json files found"
cp -r ./data* ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No data* directories found"
cp -r ./tests* ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No tests* directories found"
cp -r ./src* ../cyclonev2_simple_fresh/ 2>/dev/null || echo "No src* directories found"
echo "Files copied."

# Change to the new directory
echo "Changing to fresh directory..."
cd ../cyclonev2_simple_fresh
echo "Now in: $(pwd)"

# Initialize git
echo "Initializing git..."
git init
echo "Git initialized."

# Add files
echo "Adding files to git..."
git add .
echo "Files added."

# Commit
echo "Creating initial commit..."
git commit -m "Initial commit in fresh repository"
echo "Commit created."

echo "==== Script completed successfully ===="
echo "Your fresh repository is at: $(pwd)"
echo ""
echo "Next steps:"
echo "1. Stay in this directory: cd ../cyclonev2_simple_fresh"
echo "2. Set up Azure remote: git remote add origin https://dev.azure.com/YOUR-ORG/YOUR-PROJECT/_git/YOUR-REPO"
echo "3. Push your code: git push -u origin main"
