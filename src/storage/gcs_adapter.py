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
    def __init__(self, bucket_name: Optional[str] = None, project_id: Optional[str] = None,
                 strict: bool = False):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        )
        if not self.project_id:
            raise ValueError("Set GCP_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) before using GCS.")
        self.bucket_name = bucket_name or os.environ.get(
            "GCS_BUCKET_NAME", f"{self.project_id}-baseball-lakehouse"
        )
        self.strict = strict
        self._client = None
        self._bucket = None

    def _get_bucket(self):
        if self._bucket is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self.project_id)
                self._bucket = self._client.bucket(self.bucket_name)
            except Exception as e:
                if self.strict:
                    raise RuntimeError(
                        f"Could not connect to GCS bucket {self.bucket_name}"
                    ) from e
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
                if self.strict:
                    raise RuntimeError(f"GCS upload failed for {gcs_path}") from e
                logger.error(f"[GCS Lake] Upload failed for {gcs_path}: {e}")
        
        # Local fallback
        local_path = Path(__file__).resolve().parent.parent.parent / "data" / gcs_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_path, index=False, engine="pyarrow")
        logger.info(f"[Local Fallback] Saved to {local_path}")
        return str(local_path)

    def upload_file(self, local_path: Path, gcs_path: str) -> str:
        """Upload one existing file and fail loudly in strict cloud mode."""
        local_path = Path(local_path)
        bucket = self._get_bucket()
        if bucket is None:
            raise RuntimeError(f"GCS bucket {self.bucket_name} is unavailable")
        try:
            bucket.blob(gcs_path).upload_from_filename(str(local_path))
        except Exception as exc:
            raise RuntimeError(f"GCS upload failed for {local_path} -> {gcs_path}") from exc
        uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info("[GCS Lake] Uploaded %s to %s", local_path, uri)
        return uri

    def upload_directory(self, local_dir: Path, prefix: str) -> int:
        """Recursively upload a directory while retaining relative paths."""
        local_dir = Path(local_dir)
        if not local_dir.exists():
            return 0
        count = 0
        for local_path in sorted(path for path in local_dir.rglob("*") if path.is_file()):
            relative = local_path.relative_to(local_dir).as_posix()
            self.upload_file(local_path, f"{prefix.rstrip('/')}/{relative}")
            count += 1
        return count

    def download_directory(self, prefix: str, local_dir: Path) -> int:
        """Download every object below a prefix, returning the file count."""
        bucket = self._get_bucket()
        if bucket is None:
            raise RuntimeError(f"GCS bucket {self.bucket_name} is unavailable")
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        normalized = prefix.strip("/") + "/"
        count = 0
        for blob in self._client.list_blobs(self.bucket_name, prefix=normalized):
            if blob.name.endswith("/"):
                continue
            relative = blob.name[len(normalized):]
            if not relative or ".." in Path(relative).parts:
                raise RuntimeError(f"Unsafe GCS object path: {blob.name}")
            destination = local_dir / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))
            count += 1
        logger.info(
            "[GCS Lake] Downloaded %s files from gs://%s/%s to %s",
            count, self.bucket_name, normalized, local_dir,
        )
        return count
