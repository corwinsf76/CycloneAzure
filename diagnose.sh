#!/bin/bash

# This is a simple diagnostic script that doesn't do any Git operations
# Just to check if script execution is working

echo "======================================"
echo "Diagnostic Script for Repository Setup"
echo "======================================"
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Script location: $0"

# Check if we can write to the directory
touch test_write_access.tmp 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Can write to current directory"
    rm test_write_access.tmp
else
    echo "❌ Cannot write to current directory"
fi

# List shell scripts in current directory
echo -e "\nShell scripts in current directory:"
ls -la *.sh 2>/dev/null

echo -e "\nScript executed successfully!"
echo "If you see this message, basic script execution is working."
echo "======================================"
