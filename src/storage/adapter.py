"""
Storage Adapter for Lakehouse: Local DuckDB / Parquet & GCP BigQuery / GCS.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class StorageManager:
    """
    Manages data read/write across the 5 Medallion layers.
    Defaults to local DuckDB and Parquet files; easily configurable for GCP.
    """
    def __init__(self, base_dir: Optional[str] = None, db_path: Optional[str] = None):
        if base_dir is None:
            # default to baseball_competition directory
            self.base_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.data_dir = self.base_dir / "data"
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
