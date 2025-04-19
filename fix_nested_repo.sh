#!/bin/bash

echo "=== Fixing Nested Git Repository (cyclonev22) ==="
echo "Current directory: $(pwd)"

# Check if cyclonev22 exists and is a git repository
if [ -d "cyclonev22" ] && [ -d "cyclonev22/.git" ]; then
    echo "Found nested Git repository in cyclonev22/"
    
    echo "Options for handling the nested repository:"
    echo "1. Remove the nested .git directory (keep files, remove Git history)"
    echo "2. Remove the entire cyclonev22 directory"
    echo "3. Keep as is (not recommended)"
    
    read -p "Choose an option (1-3): " CHOICE
    
    case $CHOICE in
        1)
            echo "Removing nested Git directory..."
            git rm --cached cyclonev22
            rm -rf cyclonev22/.git
            git add cyclonev22
            git commit -m "Convert cyclonev22 from submodule to regular directory"
            echo "✅ Nested Git directory removed. cyclonev22 is now a regular directory."
            ;;
        2)
            echo "Removing entire cyclonev22 directory..."
            git rm -rf --cached cyclonev22
            rm -rf cyclonev22
            git commit -m "Remove cyclonev22 nested repository"
            echo "✅ cyclonev22 directory completely removed."
            ;;
        3)
            echo "Keeping nested repository as is."
            echo "Note: This may cause confusion in source control."
            ;;
        *)
            echo "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
    
    echo -e "\nPushing changes to Azure DevOps..."
    git push origin main
    
    echo -e "\n✅ Repository structure fixed. The changes have been pushed to Azure DevOps."
    echo "You may need to refresh your Source Control view in your editor."
else
    echo "No nested Git repository found in cyclonev22/ or the directory doesn't exist."
    
    # Check for any other directories that might be Git repositories
    echo "Checking for other potential nested Git repositories..."
    NESTED_REPOS=$(find . -name ".git" -type d | grep -v "^./.git$")
    
    if [ -n "$NESTED_REPOS" ]; then
        echo "Found the following potential nested Git repositories:"
        echo "$NESTED_REPOS"
        echo "You may want to remove these .git directories manually."
    else
        echo "No nested Git repositories found."
        echo "If you're still seeing old repository info in source control,"
        echo "try restarting your editor or refreshing the source control view."
    fi
fi
