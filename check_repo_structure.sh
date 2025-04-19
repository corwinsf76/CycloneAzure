#!/bin/bash

echo "=== Repository Structure Diagnostic ==="
echo "Current directory: $(pwd)"

# List all Git repositories in the current directory tree
echo -e "\nSearching for Git repositories..."
find . -name ".git" -type d | sort

# Check specifically for cyclonev22
if [ -d "cyclonev22" ]; then
    echo -e "\nFound cyclonev22 directory."
    if [ -d "cyclonev22/.git" ]; then
        echo "cyclonev22 is a Git repository (nested repository detected)."
        
        # Show direct commands to fix the issue
        echo -e "\nTo fix the nested repository issue, run these commands:"
        echo "1. Remove the nested repo from Git tracking:"
        echo "   git rm --cached cyclonev22"
        echo ""
        echo "2. Remove the nested .git directory:"
        echo "   rm -rf cyclonev22/.git"
        echo ""
        echo "3. Add the directory as regular files:"
        echo "   git add cyclonev22"
        echo ""
        echo "4. Commit the changes:"
        echo "   git commit -m \"Convert cyclonev22 from submodule to regular directory\""
        echo ""
        echo "5. Push to Azure DevOps:"
        echo "   git push origin main"
    else
        echo "cyclonev22 is a regular directory (good)."
    fi
else
    echo "cyclonev22 directory not found."
fi

# Check for .gitmodules file
if [ -f ".gitmodules" ]; then
    echo -e "\nFound .gitmodules file. This indicates submodules are configured."
    echo "Content of .gitmodules:"
    cat .gitmodules
    
    echo -e "\nTo remove all submodules, run:"
    echo "rm .gitmodules"
    echo "git add .gitmodules"
    echo "git commit -m \"Remove submodules configuration\""
fi

echo -e "\n=== Source Control Status ==="
echo "To refresh your source control view:"
echo "1. Close and reopen VS Code"
echo "2. Or run: code . (to restart VS Code in the current directory)"
echo "3. Or manually refresh the source control view in your editor"
