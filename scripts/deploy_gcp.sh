#!/usr/bin/env bash
set -euo pipefail

# Run this from Google Cloud Shell after cloning/pulling the GitHub repository.
# It deploys manual-only jobs; add --run to execute prepare followed by evaluate.
RUN_JOBS=0
if [[ "${1:-}" == "--run" ]]; then
  RUN_JOBS=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/deploy_gcp.sh [--run]" >&2
  exit 2
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
DATASET_ID="${BIGQUERY_DATASET:-baseball_analytics}"
BUCKET_NAME="${GCS_BUCKET_NAME:-${PROJECT_ID}-baseball-lakehouse}"
REPOSITORY="${ARTIFACT_REPOSITORY:-baseball-pipeline}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/pipeline:latest"
SERVICE_ACCOUNT_NAME="baseball-pipeline-runner"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Select a project first: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 2
fi

echo "Deploying to project=${PROJECT_ID}, region=${REGION}, bucket=${BUCKET_NAME}"
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com bigquery.googleapis.com --project="${PROJECT_ID}"

if ! gcloud artifacts repositories describe "${REPOSITORY}" \
  --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" --repository-format=docker \
    --location="${REGION}" --project="${PROJECT_ID}"
fi

if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" \
    --location="${REGION}" --uniform-bucket-level-access
fi

if ! bq --project_id="${PROJECT_ID}" show --dataset \
  "${PROJECT_ID}:${DATASET_ID}" >/dev/null 2>&1; then
  bq --project_id="${PROJECT_ID}" --location="US" mk --dataset "${DATASET_ID}"
fi

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --display-name="Baseball pipeline Cloud Run jobs" --project="${PROJECT_ID}"
fi

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" --role="roles/storage.objectAdmin" \
  --project="${PROJECT_ID}" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" --role="roles/bigquery.dataEditor" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" --role="roles/bigquery.jobUser" >/dev/null

gcloud builds submit --project="${PROJECT_ID}" --tag="${IMAGE}" .

COMMON_ENV="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET_NAME=${BUCKET_NAME},BIGQUERY_DATASET=${DATASET_ID},PYTHONUNBUFFERED=1"
gcloud run jobs deploy baseball-prepare \
  --image="${IMAGE}" --region="${REGION}" --project="${PROJECT_ID}" \
  --command=python --args=scripts/cloud_entrypoint.py \
  --service-account="${SERVICE_ACCOUNT}" --tasks=1 --parallelism=1 \
  --cpu=1 --memory=4Gi --task-timeout=7200s --max-retries=1 \
  --set-env-vars="${COMMON_ENV},PIPELINE_STAGE=prepare"

gcloud run jobs deploy baseball-evaluate \
  --image="${IMAGE}" --region="${REGION}" --project="${PROJECT_ID}" \
  --command=python --args=scripts/cloud_entrypoint.py \
  --service-account="${SERVICE_ACCOUNT}" --tasks=1 --parallelism=1 \
  --cpu=2 --memory=8Gi --task-timeout=21600s --max-retries=0 \
  --set-env-vars="${COMMON_ENV},PIPELINE_STAGE=evaluate,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2"

echo "Deployment complete. These jobs run only when explicitly executed."
echo "Prepare:  gcloud run jobs execute baseball-prepare --region=${REGION} --wait"
echo "Evaluate: gcloud run jobs execute baseball-evaluate --region=${REGION} --wait"

if [[ "${RUN_JOBS}" -eq 1 ]]; then
  gcloud run jobs execute baseball-prepare --region="${REGION}" \
    --project="${PROJECT_ID}" --wait
  gcloud run jobs execute baseball-evaluate --region="${REGION}" \
    --project="${PROJECT_ID}" --wait
fi
