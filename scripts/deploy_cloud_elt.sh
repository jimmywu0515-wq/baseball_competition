#!/bin/bash
set -e

# ==============================================================================
# Deploy Full Cloud ELT Workflow to Google Cloud Platform
# Target: Cloud Run Jobs (Serverless Batch Pipeline) + Cloud Storage + BigQuery
# ==============================================================================

PROJECT_ID=${GCP_PROJECT_ID:-"project-f677f84f-db22-4976-96b"}
REGION=${GCP_REGION:-"us-central1"}
BUCKET_NAME=${GCS_BUCKET_NAME:-"${PROJECT_ID}-baseball-lakehouse"}
DATASET_NAME=${BIGQUERY_DATASET:-"baseball_analytics"}
JOB_NAME="baseball-elt-pipeline-job"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${JOB_NAME}:latest"

echo "========================================================"
echo "DEPLOYING FULL CLOUD ELT WORKFLOW TO GCP"
echo "Project: $PROJECT_ID | Region: $REGION"
echo "Lake Bucket: gs://$BUCKET_NAME | Warehouse: $DATASET_NAME"
echo "========================================================"

# 1. Ensure GCP project is set
gcloud config set project "$PROJECT_ID"

# 2. Enable Required Cloud APIs
echo "[1/5] Enabling GCP APIs (BigQuery, Storage, Cloud Run, Cloud Build)..."
gcloud services enable \
    bigquery.googleapis.com \
    storage.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID"

# 3. Create Cloud Storage Data Lake Bucket if not exists
echo "[2/5] Creating GCS Data Lake Bucket..."
gcloud storage buckets create "gs://$BUCKET_NAME" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access 2>/dev/null || echo "Bucket gs://$BUCKET_NAME already exists."

# 4. Initialize BigQuery Dataset & 5-Layer Table Schemas
echo "[3/5] Initializing BigQuery Dataset & Tables..."
bq --location="$REGION" mk --dataset --default_table_expiration 0 "$PROJECT_ID:$DATASET_NAME" 2>/dev/null || echo "Dataset $DATASET_NAME exists."
bq query --use_legacy_sql=false --project_id="$PROJECT_ID" < "$(dirname "$0")/init_bigquery.sql"

# 5. Build Cloud ELT Container Image using Cloud Build (No local Docker required!)
echo "[4/5] Submitting Container Build to Cloud Build..."
gcloud builds submit --tag "$IMAGE_NAME" .

# 6. Deploy as a Serverless Cloud Run Job
echo "[5/5] Creating/Updating Cloud Run Job '$JOB_NAME'..."
gcloud run jobs deploy "$JOB_NAME" \
    --image="$IMAGE_NAME" \
    --region="$REGION" \
    --command="python" \
    --args="scripts/run_cloud_elt.py" \
    --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCS_BUCKET_NAME=$BUCKET_NAME,BIGQUERY_DATASET=$DATASET_NAME" \
    --memory=4Gi \
    --cpu=2 \
    --max-retries=1

echo "========================================================"
echo "DEPLOYMENT COMPLETE!"
echo ""
echo "To trigger the ELT pipeline immediately on GCP, run:"
echo "  gcloud run jobs execute $JOB_NAME --region=$REGION"
echo ""
echo "Or click 'Execute' directly in Google Cloud Console > Cloud Run > Jobs."
echo "========================================================"
