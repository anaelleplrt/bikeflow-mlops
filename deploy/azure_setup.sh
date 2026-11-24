#!/usr/bin/env bash
# One-time Azure setup for the BikeFlow production service.
#
# Prerequisites: Azure CLI, logged in with `az login` on the subscription
# that holds the DVC storage account.
#
# Creates (if missing) the Container Apps environment in the existing resource
# group and a service principal limited to that resource group, used by the
# GitHub Actions deploy workflow. The Container App itself is created by the
# first run of .github/workflows/deploy.yml from deploy/containerapp.yaml.
#
# Usage: bash deploy/azure_setup.sh

set -euo pipefail

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-bikeflow-rg}"
LOCATION="${AZURE_LOCATION:-francecentral}"
ENVIRONMENT="${AZURE_CONTAINER_ENVIRONMENT:-bikeflow-env}"
SP_NAME="${AZURE_SP_NAME:-bikeflow-github-deploy}"

echo "Subscription: $(az account show --query name -o tsv)"
echo "Resource group: $RESOURCE_GROUP ($LOCATION)"

az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az group show --name "$RESOURCE_GROUP" --output none

if az containerapp env show --name "$ENVIRONMENT" --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null; then
  echo "Environment $ENVIRONMENT already exists"
else
  # Logs go to the Container Apps console only: no Log Analytics workspace to pay for.
  az containerapp env create \
    --name "$ENVIRONMENT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --logs-destination none \
    --output none
  echo "Environment $ENVIRONMENT created"
fi

scope=$(az group show --name "$RESOURCE_GROUP" --query id -o tsv)
echo
echo "Service principal for GitHub Actions (Contributor on $RESOURCE_GROUP only)."
echo "Copy the JSON below into the GitHub secret AZURE_CREDENTIALS, then clear your terminal:"
echo
az ad sp create-for-rbac --name "$SP_NAME" --role contributor --scopes "$scope" --json-auth

cat <<EOF

GitHub repository variables to set (Settings > Secrets and variables > Actions > Variables):
  AZURE_RESOURCE_GROUP=$RESOURCE_GROUP
  AZURE_LOCATION=$LOCATION
  AZURE_CONTAINER_ENVIRONMENT=$ENVIRONMENT
  AZURE_CONTAINER_APP=bikeflow
  MLFLOW_TRACKING_URI=<DagsHub MLflow URI>
EOF
