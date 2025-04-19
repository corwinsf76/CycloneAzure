#!/bin/bash

echo "=== Cleaning up temporary scripts ==="
echo "Removing fix scripts and temporary files..."

# List of scripts to remove
SCRIPTS=(
  "azure_auth.sh"
  "azure_auth_simple.sh"
  "azure_connect.sh"
  "azure_connect_simple.sh"
  "azure_guide.sh"
  "azure_remote_setup.sh"
  "azure_setup_local.sh"
  "check_repository_status.sh"
  "create_azure_repo.sh"
  "debug_repo.sh"
  "diagnose.sh"
  "direct_push.sh"
  "fix_git_config.sh"
  "fix_repository.sh"
  "fix_url.sh"
  "fresh_repo.sh"
  "git_reset.sh"
  "local_azure_setup.sh"
  "origin.bak"
  "push_to_azure.sh"
  "reinit_git.sh"
  "setup_azure_repo.sh"
  "simple_fresh_repo.sh"
  "simple_push.sh"
  "switch_repos.sh"
  "test_script.sh"
  "update_remote.sh"
  "vs_code_azure_setup.sh"
  "check_repo_structure.sh"
  "direct_fix_commands.txt"
)

# Remove each script
for script in "${SCRIPTS[@]}"; do
  if [ -f "$script" ]; then
    rm "$script"
    echo "Removed: $script"
  fi
done

echo "Creating a reference guide for Azure DevOps..."

# Create a reference guide for future use
cat > azure_devops_guide.md << 'EOF'
# Azure DevOps Git Reference Guide

## Repository Information
- Repository URL: https://dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2
- Local directory: /home/Justin/projects/cyclonev2

## Common Git Commands

### Basic Commands
```bash
# Pull latest changes
git pull origin main

# Create a new branch
git checkout -b feature/new-feature

# Stage changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push changes
git push origin main
```

### Advanced Commands
```bash
# Create a tag (for releases)
git tag v1.0.0
git push origin v1.0.0

# View commit history
git log --oneline --graph

# Create a patch
git diff > changes.patch

# Apply a patch
git apply changes.patch
```

## Azure DevOps Features

- **Boards**: Track work with Kanban boards
- **Repos**: Manage source code
- **Pipelines**: Set up CI/CD pipelines
- **Test Plans**: Create and run tests
- **Artifacts**: Manage packages

## Authentication

If you need to update your credentials:
1. Create a new Personal Access Token (PAT) at:
   https://dev.azure.com/justinlaughlin/_usersSettings/tokens
2. Use this token as your password when Git prompts for authentication

## Best Practices

1. Pull before pushing to avoid merge conflicts
2. Use descriptive commit messages
3. Create feature branches for new work
4. Use meaningful tags for releases
5. Keep commits focused on a single task
EOF

chmod +x cleanup.sh
echo "✅ Cleanup script created."
echo "Reference guide created: azure_devops_guide.md"

echo -e "\n=== Your repository is now properly set up with Azure DevOps ==="
echo "Your code is available at: https://dev.azure.com/justinlaughlin/cyclonev2/_git/cyclonev2"
echo "Local repository: $(pwd)"
echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"
