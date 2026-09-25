"""
Storage Adapter for Lakehouse: Local DuckDB / Parquet & GCP BigQuery / GCS.
"""
import os
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence, Tuple
import pandas as pd
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class StorageManager:
    """
    Manages data read/write across the 5 Medallion layers.
    Defaults to local DuckDB and Parquet files; easily configurable for GCP.
    """
    def __init__(self, base_dir: Optional[str] = None, db_path: Optional[str] = None,
                 namespace: Optional[str] = None):
        if base_dir is None:
            # default to baseball_competition directory
            self.base_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        if namespace is not None and (not namespace.isidentifier() or namespace == "real"):
            raise ValueError("Storage namespace must be a simple non-real identifier")
        self.data_dir = self.base_dir / "data" / namespace if namespace else self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.silver_dir = self.data_dir / "silver"
        self.gold_dir = self.data_dir / "gold"
        
        # Ensure directories exist
        for d in [self.raw_dir, self.silver_dir, self.gold_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            self.db_path = str(self.data_dir / "baseball_warehouse.duckdb")
        else:
            self.db_path = db_path
            
        self.conn = duckdb.connect(self.db_path)
        logger.info(f"Initialized DuckDB Lakehouse at {self.db_path}")

    def save_table(self, df: pd.DataFrame, table_name: str, layer: str = "silver", partition_cols: Optional[List[str]] = None) -> str:
        """
        Saves a DataFrame both to a DuckDB table and as a Parquet dataset.
        layer: 'raw', 'silver', or 'gold'
        """
        if df.empty:
            logger.warning(f"DataFrame for {table_name} is empty. Skipping save.")
            return ""

        # 1. Save / Replace DuckDB Table
        self.conn.register("temp_df", df)
        self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df")
        self.conn.unregister("temp_df")

        # 2. Save Parquet
        target_dir = getattr(self, f"{layer}_dir", self.silver_dir)
        file_path = target_dir / f"{table_name}.parquet"
        df.to_parquet(file_path, index=False, engine="pyarrow")
        logger.info(f"Saved {len(df)} rows to DuckDB table '{table_name}' and Parquet file '{file_path}'")
        return str(file_path)

    def save_tables_atomically(
        self,
        tables: Sequence[Tuple[pd.DataFrame, str, str]],
    ) -> Dict[str, str]:
        """Publish a complete warehouse batch or restore every previous table/file.

        Parquet files are prepared before the DuckDB transaction begins. Existing
        Parquet files are retained as temporary backups until the database commit
        succeeds, preventing a partial pipeline run from mixing old and new layers.
        """
        token = uuid.uuid4().hex
        prepared = []
        registered = []
        for frame, table_name, layer in tables:
            target_dir = getattr(self, f"{layer}_dir", self.silver_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            final_path = target_dir / f"{table_name}.parquet"
            temp_path = target_dir / f".{table_name}.{token}.tmp.parquet"
            backup_path = target_dir / f".{table_name}.{token}.bak.parquet"
            frame.to_parquet(temp_path, index=False, engine="pyarrow")
            prepared.append({
                "frame": frame, "name": table_name, "final": final_path,
                "temp": temp_path, "backup": backup_path,
                "had_original": final_path.exists(), "installed": False,
            })

        self.conn.execute("BEGIN TRANSACTION")
        try:
            for position, item in enumerate(prepared):
                registration = f"batch_df_{position}_{token}"
                self.conn.register(registration, item["frame"])
                registered.append(registration)
                self.conn.execute(
                    f'CREATE OR REPLACE TABLE "{item["name"]}" AS SELECT * FROM "{registration}"'
                )

            for item in prepared:
                if item["had_original"]:
                    os.replace(item["final"], item["backup"])
                os.replace(item["temp"], item["final"])
                item["installed"] = True

            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            for item in reversed(prepared):
                if item["installed"] and item["final"].exists():
                    item["final"].unlink()
                if item["backup"].exists():
                    os.replace(item["backup"], item["final"])
                if item["temp"].exists():
                    item["temp"].unlink()
            raise
        finally:
            for registration in registered:
                try:
                    self.conn.unregister(registration)
                except Exception:
                    pass

        for item in prepared:
            if item["backup"].exists():
                item["backup"].unlink()
            logger.info(
                "Atomically published %s rows to '%s' and '%s'",
                len(item["frame"]), item["name"], item["final"],
            )
        return {item["name"]: str(item["final"]) for item in prepared}

    def load_table(self, table_name: str, layer: Optional[str] = None) -> pd.DataFrame:
        """
        Loads a table from DuckDB or Parquet.
        """
        try:
            return self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        except Exception as e:
            # Fallback to parquet
            if layer:
                target_dir = getattr(self, f"{layer}_dir", self.silver_dir)
                file_path = target_dir / f"{table_name}.parquet"
                if file_path.exists():
                    return pd.read_parquet(file_path)
            # Try searching all layers
            for l in ["raw", "silver", "gold"]:
                p = getattr(self, f"{l}_dir") / f"{table_name}.parquet"
                if p.exists():
                    return pd.read_parquet(p)
            logger.warning(f"Table or Parquet '{table_name}' not found: {e}")
            return pd.DataFrame()

    def query(self, sql: str) -> pd.DataFrame:
        """Executes analytical SQL directly on the local DuckDB warehouse."""
        return self.conn.execute(sql).fetchdf()

    def close(self):
        self.conn.close()
