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
