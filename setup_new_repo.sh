#!/bin/bash

echo "Setting up new repository for your codebase"

# 1. Remove existing Git repository metadata
if [ -d ".git" ]; then
  echo "Removing existing Git repository..."
  rm -rf .git
  echo "Previous Git repository removed."
else
  echo "No existing Git repository found."
fi

# 2. Initialize a new Git repository
echo "Initializing new Git repository..."
git init

# 3. Add all files to the new repository
echo "Adding files to the repository..."
git add .

# 4. Make initial commit
echo "Creating initial commit..."
git commit -m "Initial commit for cyclonev2 project"

echo "✅ Local repository created successfully!"
echo
echo "Next steps:"
echo "1. Create a new repository on GitHub/Azure DevOps/GitLab"
echo "2. Connect to remote repository with:"
echo "   git remote add origin <new-repository-url>"
echo "3. Push your code with:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "For Azure CI/CD, update your pipeline configurations to point to the new repository."
