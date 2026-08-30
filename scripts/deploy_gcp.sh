#!/bin/bash
set -e

# GCP Deployment Script for Baseball Lakehouse & Dashboard
# Ensure gcloud CLI is authenticated and active

PROJECT_ID=${GCP_PROJECT_ID:-"your-gcp-project-id"}
REGION=${GCP_REGION:-"us-central1"}
BUCKET_NAME=${GCS_BUCKET_NAME:-"${PROJECT_ID}-baseball-lakehouse"}
DATASET_NAME=${BIGQUERY_DATASET:-"baseball_analytics"}
SERVICE_NAME="baseball-fatigue-dashboard"

echo "========================================================"
echo "Deploying Baseball Fatigue System to GCP"
echo "Project: $PROJECT_ID | Region: $REGION | Bucket: $BUCKET_NAME"
echo "========================================================"

# 1. Enable GCP Services
echo "Enabling Google Cloud APIs..."
gcloud services enable \
    bigquery.googleapis.com \
    storage.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID"

# 2. Create GCS Bucket for Parquet Lakehouse
echo "Creating GCS Bucket if not exists..."
gcloud storage buckets create "gs://$BUCKET_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access || true

# 3. Create BigQuery Dataset & Tables
echo "Initializing BigQuery Dataset and Tables..."
bq --location="$REGION" mk --dataset --default_table_expiration 0 "$PROJECT_ID:$DATASET_NAME" || true
bq query --use_legacy_sql=false --project_id="$PROJECT_ID" < "$(dirname "$0")/init_bigquery.sql"

# 4. Build and Deploy Cloud Run Service
echo "Building and Deploying Coach Dashboard to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --source=. \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCS_BUCKET_NAME=$BUCKET_NAME,BIGQUERY_DATASET=$DATASET_NAME"

echo "========================================================"
echo "Deployment Completed Successfully!"
echo "========================================================"
