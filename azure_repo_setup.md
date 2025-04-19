# Azure Repository Setup Guide

This guide helps you set up your codebase with Azure DevOps repositories after removing a previous repository connection.

## Creating a New Azure DevOps Repository

1. Log into [Azure DevOps](https://dev.azure.com)
2. Navigate to your organization and project (or create a new one)
3. Go to Repos > Files
4. If you need a new repository, click on the repository dropdown and select "New repository"
5. Enter a name for your repository (e.g., "cyclonev2")
6. Click "Create"

## Connecting Your Local Code to Azure Repos

From your local machine:

```bash
# Make sure you're in the project directory
cd /home/Justin/projects/cyclonev2

# Add the Azure repository as a remote
git remote add origin https://dev.azure.com/YOUR-ORG/YOUR-PROJECT/_git/YOUR-REPO

# Push your code
git push -u origin main
```

## Updating Azure Pipelines

If you have existing pipelines that were connected to your old repository:

1. Go to Pipelines > Pipelines
2. Edit each pipeline
3. Under "Get Sources", select your new repository
4. Save the pipeline

## Managing Azure Resources Connected to Your Repo

For Azure resources that were deployed from your repository:

1. Update deployment sources in Azure Portal for any App Services
2. Update Azure DevOps Release Pipelines to use the new repository
3. Re-configure any GitHub Actions or Azure DevOps webhooks

## Breaking Connections with Old Repository

1. Check your Azure resources for any connections to the old repository
2. Update all deployment sources in Azure Portal
3. Remove any personal access tokens (PATs) associated with the old repository
4. Update any continuous deployment configurations
