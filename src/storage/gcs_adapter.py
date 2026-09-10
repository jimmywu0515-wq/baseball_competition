"""
GCS Data Lake Adapter: Handles streaming Parquet files directly to Google Cloud Storage.
"""
import os
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class GCSLakeManager:
    """
    Manages direct streaming and reading of Data Lake Parquet files on GCS.
    Falls back to local file system if running in offline/local mode.
    """
    def __init__(self, bucket_name: Optional[str] = None, project_id: Optional[str] = None):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "project-f677f84f-db22-4976-96b")
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME", f"{self.project_id}-baseball-lakehouse")
        self._client = None
        self._bucket = None

    def _get_bucket(self):
        if self._bucket is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self.project_id)
                self._bucket = self._client.bucket(self.bucket_name)
            except Exception as e:
                logger.warning(f"Could not connect to GCS bucket {self.bucket_name}: {e}")
        return self._bucket

    def upload_parquet_to_lake(self, df: pd.DataFrame, gcs_path: str) -> str:
        """
        Streams a DataFrame as a Parquet file directly to GCS Data Lake.
        Example gcs_path: 'raw/statcast_pitches/season=2024/data.parquet'
        """
        bucket = self._get_bucket()
        if bucket is not None:
            try:
                buffer = BytesIO()
                df.to_parquet(buffer, index=False, engine="pyarrow")
                buffer.seek(0)
                blob = bucket.blob(gcs_path)
                blob.upload_from_file(buffer, content_type="application/octet-stream")
                uri = f"gs://{self.bucket_name}/{gcs_path}"
                logger.info(f"[GCS Lake] Uploaded {len(df)} rows to {uri}")
                return uri
            except Exception as e:
                logger.error(f"[GCS Lake] Upload failed for {gcs_path}: {e}")
        
        # Local fallback
        local_path = Path(__file__).resolve().parent.parent.parent / "data" / gcs_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_path, index=False, engine="pyarrow")
        logger.info(f"[Local Fallback] Saved to {local_path}")
        return str(local_path)
