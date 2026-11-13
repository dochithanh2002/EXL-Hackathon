#!/usr/bin/env bash
# One-time Azure ML workspace setup. Run after `az login`.
set -euo pipefail
source agent/config/azure.env
az account set --subscription "$AZURE_SUBSCRIPTION_ID"
az extension add -n ml --upgrade -y
az group create -n "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION"
az ml workspace create -n "$AZURE_ML_WORKSPACE" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION"
echo "Workspace ready. MLflow URI:"
az ml workspace show -n "$AZURE_ML_WORKSPACE" -g "$AZURE_RESOURCE_GROUP" --query mlflow_tracking_uri -o tsv
