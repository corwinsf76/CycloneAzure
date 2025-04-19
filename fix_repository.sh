#!/bin/bash

echo "=== Fixing repository issues before connecting to Azure ==="

# 1. Handle the embedded git repository issue (cyclonev22)
if [ -d "cyclonev22" ] && [ -d "cyclonev22/.git" ]; then
    echo "Found embedded Git repository at cyclonev22/"
    read -p "Do you want to: (1) Remove it, (2) Convert to a regular directory, or (3) Keep as submodule? [1/2/3]: " REPO_CHOICE
    
    case $REPO_CHOICE in
        1)
            echo "Removing cyclonev22 directory entirely..."
            git rm --cached cyclonev22
            rm -rf cyclonev22
            echo "cyclonev22 directory removed."
            ;;
        2)
            echo "Converting cyclonev22 to a regular directory..."
            git rm --cached cyclonev22
            rm -rf cyclonev22/.git
            git add cyclonev22
            echo "cyclonev22 converted to a regular directory."
            ;;
        3)
            echo "Keeping as a submodule. You'll need to properly initialize it later."
            echo "If you have the URL, you can run: git submodule add <url> cyclonev22"
            ;;
        *)
            echo "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
    
    # Commit the changes
    git commit -m "Fix cyclonev22 embedded repository issue"
else
    echo "No embedded Git repository found at cyclonev22/ or directory doesn't exist."
    echo "Skipping this step."
fi

# 2. Rename branch from master to main
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" = "master" ]; then
    echo "Renaming branch from 'master' to 'main'..."
    git branch -m master main
    echo "Branch renamed to 'main'."
else
    echo "Current branch is $CURRENT_BRANCH. No need to rename."
fi

# 3. Set the default branch for new repositories
echo "Setting the default branch name for new repositories to 'main'..."
git config --global init.defaultBranch main
echo "Default branch set to 'main'."

echo "✅ Repository fixes completed!"
echo
echo "Next steps:"
echo "1. Run ./check_repository_status.sh to verify your repository is ready"
echo "2. Run ./connect_to_azure.sh to connect to Azure DevOps"
#!/bin/bash

echo "Fixing repository issues before connecting to Azure..."

# 1. Handle the embedded git repository issue (cyclonev22)
if [ -d "cyclonev22" ] && [ -d "cyclonev22/.git" ]; then
    echo "Found embedded Git repository at cyclonev22/"
    read -p "Do you want to: (1) Remove it, (2) Convert to a regular directory, or (3) Keep as submodule? [1/2/3]: " REPO_CHOICE
    
    case $REPO_CHOICE in
        1)
            echo "Removing cyclonev22 directory entirely..."
            git rm --cached cyclonev22
            rm -rf cyclonev22
            echo "cyclonev22 directory removed."
            ;;
        2)
            echo "Converting cyclonev22 to a regular directory..."
            git rm --cached cyclonev22
            rm -rf cyclonev22/.git
            git add cyclonev22
            echo "cyclonev22 converted to a regular directory."
            ;;
        3)
            echo "Keeping as a submodule. You'll need to properly initialize it later."
            echo "If you have the URL, you can run: git submodule add <url> cyclonev22"
            ;;
        *)
            echo "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
    
    # Commit the changes
    git commit -m "Fix cyclonev22 embedded repository issue"
fi

# 2. Rename branch from master to main
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" = "master" ]; then
    echo "Renaming branch from 'master' to 'main'..."
    git branch -m master main
    echo "Branch renamed to 'main'."
else
    echo "Current branch is $CURRENT_BRANCH. No need to rename."
fi

# 3. Set the default branch for new repositories
echo "Setting the default branch name for new repositories to 'main'..."
git config --global init.defaultBranch main
echo "Default branch set to 'main'."

echo "✅ Repository fixes completed!"
echo "You can now proceed with connecting to Azure DevOps using ./connect_to_azure.sh"
