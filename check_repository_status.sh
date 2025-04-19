#!/bin/bash

echo "=== Cyclone v2 Repository Status Check ==="
echo "Checking your Git repository status..."

# Check if we're in a Git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a Git repository root directory."
    echo "   Current directory: $(pwd)"
    echo "   Please navigate to your repository root or initialize Git."
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Check if the cyclonev22 directory exists
if [ -d "cyclonev22" ]; then
    echo "Found cyclonev22 directory"
    
    # Check if it's a Git repository
    if [ -d "cyclonev22/.git" ]; then
        echo "❗ cyclonev22 is an embedded Git repository (submodule issue)"
        echo "   To fix this, run ./fix_repository.sh and select option 2"
    else
        echo "✅ cyclonev22 is a regular directory (good)"
    fi
else
    echo "cyclonev22 directory not found"
fi

# Check for Git remotes
echo -e "\nConfigured remotes:"
git remote -v
if [ $? -ne 0 ] || [ -z "$(git remote)" ]; then
    echo "❗ No remotes configured. You'll need to add an Azure DevOps remote."
    echo "   Run ./connect_to_azure.sh to set up an Azure DevOps remote."
fi

# Check Git status
echo -e "\nGit status:"
git status --short

# Check if fix_repository.sh exists and is executable
if [ -f "fix_repository.sh" ]; then
    if [ -x "fix_repository.sh" ]; then
        echo "✅ fix_repository.sh exists and is executable"
    else
        echo "❗ fix_repository.sh exists but is not executable"
        echo "   Run: chmod +x fix_repository.sh"
    fi
else
    echo "❌ fix_repository.sh not found"
fi

# Check if connect_to_azure.sh exists and is executable
if [ -f "connect_to_azure.sh" ]; then
    if [ -x "connect_to_azure.sh" ]; then
        echo "✅ connect_to_azure.sh exists and is executable"
    else
        echo "❗ connect_to_azure.sh exists but is not executable"
        echo "   Run: chmod +x connect_to_azure.sh"
    fi
else
    echo "❌ connect_to_azure.sh not found"
fi

echo -e "\n=== Next Steps ==="
echo "1. If cyclonev22 is an embedded Git repository, run ./fix_repository.sh"
echo "2. To connect to Azure DevOps, run ./connect_to_azure.sh"
echo "3. If you need to push to Azure DevOps, make sure your branch is 'main'"
echo "   (use 'git branch -m master main' to rename if needed)"
