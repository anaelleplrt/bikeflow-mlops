#!/usr/bin/env bash
# One-time Azure setup for the BikeFlow production service.
#
# Prerequisites: Azure CLI, logged in with `az login` on the subscription
# that holds the DVC storage account.
#
# Creates (if missing) the Container Apps environment in the existing resource
# group and a user-assigned managed identity that GitHub Actions signs in as
# through OIDC: no client secret is created or stored anywhere. The identity is
# Contributor on the resource group only. The Container App itself is created
# by the first run of .github/workflows/deploy.yml from deploy/containerapp.yaml.
#
# A managed identity needs no directory permission (the EFREI directory does
# not let students register applications, so a service principal is not an
# option). Requires Owner, or Contributor + Role Based Access Control
# Administrator, on the resource group to assign the identity its role.
#
# Usage: AZURE_SUBSCRIPTION="<subscription name or id>" bash deploy/azure_setup.sh

set -euo pipefail

# Git Bash on Windows rewrites arguments starting with "/" (such as the
# /subscriptions/... role scope) into Windows paths. No effect elsewhere.
export MSYS_NO_PATHCONV=1

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-bikeflow-rg}"
LOCATION="${AZURE_LOCATION:-francecentral}"
ENVIRONMENT="${AZURE_CONTAINER_ENVIRONMENT:-bikeflow-env}"
IDENTITY="${AZURE_IDENTITY:-bikeflow-github-deploy}"
REPOSITORY="${GITHUB_REPOSITORY:-anaelleplrt/bikeflow-mlops}"

# The resources live in the subscription that holds the DVC storage account.
# Pass it explicitly when your login sees several subscriptions.
if [ -n "${AZURE_SUBSCRIPTION:-}" ]; then
  az account set --subscription "$AZURE_SUBSCRIPTION"
fi

echo "Subscription: $(az account show --query name -o tsv)"
echo "Resource group: $RESOURCE_GROUP ($LOCATION)"

if ! az group show --name "$RESOURCE_GROUP" --output none 2>/dev/null; then
  echo "Resource group $RESOURCE_GROUP not found in this subscription." >&2
  echo "Run 'az account list -o table' and set AZURE_SUBSCRIPTION, or ask the owner for access." >&2
  exit 1
fi

az extension add --name containerapp --upgrade --only-show-errors
# Registering a provider needs rights on the whole subscription: only try when needed.
for namespace in Microsoft.App Microsoft.OperationalInsights Microsoft.ManagedIdentity; do
  state=$(az provider show --namespace "$namespace" --query registrationState -o tsv 2>/dev/null || true)
  if [ "$state" != "Registered" ]; then
    az provider register --namespace "$namespace" --wait || {
      echo "Cannot register $namespace. Ask the subscription owner to run:" >&2
      echo "  az provider register --namespace $namespace" >&2
      exit 1
    }
  fi
done

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

if az identity show --name "$IDENTITY" --resource-group "$RESOURCE_GROUP" --output none 2>/dev/null; then
  echo "Managed identity $IDENTITY already exists"
else
  az identity create --name "$IDENTITY" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" --output none
  echo "Managed identity $IDENTITY created"
fi
client_id=$(az identity show --name "$IDENTITY" --resource-group "$RESOURCE_GROUP" --query clientId -o tsv)
principal_id=$(az identity show --name "$IDENTITY" --resource-group "$RESOURCE_GROUP" --query principalId -o tsv)

# Trust GitHub tokens from this repository only: the deploy job (environment
# production) and workflows run from main (continuous training, rollback).
# GitHub's OIDC subject names the owner and repository with their numeric ids
# (owner@id/repo@id); both ids are on https://api.github.com/repos/<owner>/<repo>.
# create also updates an existing credential, so the script can be re-run.
SUBJECT_PREFIX="${GITHUB_OIDC_SUBJECT_PREFIX:-repo:anaelleplrt@128377927/bikeflow-mlops@1464646886}"
for entry in "github-production|$SUBJECT_PREFIX:environment:production" \
             "github-main|$SUBJECT_PREFIX:ref:refs/heads/main"; do
  name="${entry%%|*}"
  subject="${entry#*|}"
  az identity federated-credential create --name "$name" --identity-name "$IDENTITY" \
    --resource-group "$RESOURCE_GROUP" \
    --issuer https://token.actions.githubusercontent.com \
    --subject "$subject" --audiences api://AzureADTokenExchange --output none
  echo "Federated credential $name trusts $subject"
done

# Contributor on the resource group only. A new identity can take a minute to
# be visible to role assignments, hence the retries.
existing=$(az role assignment list --assignee "$principal_id" --scope "$scope" --role Contributor --query "[0].id" -o tsv 2>/dev/null || true)
if [ -n "$existing" ]; then
  echo "$IDENTITY is already Contributor on $RESOURCE_GROUP"
else
  for attempt in 1 2 3 4 5 6; do
    if az role assignment create --assignee-object-id "$principal_id" --assignee-principal-type ServicePrincipal \
         --role Contributor --scope "$scope" --output none; then
      echo "Contributor on $RESOURCE_GROUP granted to $IDENTITY"
      break
    fi
    if [ "$attempt" = 6 ]; then
      echo "Role assignment failed" >&2
      exit 1
    fi
    sleep 10
  done
fi

tenant_id=$(az account show --query tenantId -o tsv)
subscription_id=$(az account show --query id -o tsv)

cat <<VARIABLES

GitHub repository variables to set (Settings > Secrets and variables > Actions > Variables).
None of them is a secret: the identity only accepts tokens GitHub issues for $REPOSITORY.
  AZURE_CLIENT_ID=$client_id
  AZURE_TENANT_ID=$tenant_id
  AZURE_SUBSCRIPTION_ID=$subscription_id
  AZURE_RESOURCE_GROUP=$RESOURCE_GROUP
  AZURE_LOCATION=$LOCATION
  AZURE_CONTAINER_ENVIRONMENT=$ENVIRONMENT
  AZURE_CONTAINER_APP=bikeflow
  MLFLOW_TRACKING_URI=<DagsHub MLflow URI>
VARIABLES
