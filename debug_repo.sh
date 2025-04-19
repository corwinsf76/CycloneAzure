#!/bin/bash

echo "=== Repository Debug Tool ==="
echo "Current directory: $(pwd)"
echo "Checking for Git repository..."

# Basic checks
if [ -d ".git" ]; then
  echo "✅ Git repository found in current directory"
else
  echo "❌ No Git repository found in current directory"
fi

# Show directories and key files
echo -e "\nDirectory contents:"
ls -la

# Try running git commands
echo -e "\nGit status:"
git status

# Check for specific files
echo -e "\nChecking specific directories and files:"
for item in "cyclonev22" ".git" "connect_to_azure.sh" "fix_repository.sh" "check_repository_status.sh"; do
  if [ -e "$item" ]; then
    if [ -d "$item" ]; then
      echo "✅ $item exists (directory)"
    else
      if [ -x "$item" ]; then
        echo "✅ $item exists (executable file)"
      else
        echo "⚠️ $item exists (file, not executable)"
      fi
    fi
  else
    echo "❌ $item not found"
  fi
done

# Check script paths
echo -e "\nFull paths:"
echo "check_repository_status.sh: $(readlink -f check_repository_status.sh 2>/dev/null || echo 'not found')"
echo "fix_repository.sh: $(readlink -f fix_repository.sh 2>/dev/null || echo 'not found')"
echo "connect_to_azure.sh: $(readlink -f connect_to_azure.sh 2>/dev/null || echo 'not found')"

echo -e "\n=== Actions to Take ==="
echo "1. Make scripts executable: chmod +x *.sh"
echo "2. Run scripts from the repository root directory"
echo "3. Try using relative paths: ./script_name.sh"
