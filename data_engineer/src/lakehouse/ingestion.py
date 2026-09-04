"""
Ingestion Engine for Lakehouse Bronze Layer.
Reads raw batches, adds audit metadata, and writes append-only Parquet files.
"""

import datetime
import hashlib
from pathlib import Path
from typing import Dict, Optional, Union
import uuid
import pandas as pd
from ..config import LakehouseConfig, load_lakehouse_config
from .contracts import validate_bronze_data
from .db import DuckDBManager


class LakehouseIngestion:
    def __init__(self, config: Optional[LakehouseConfig] = None):
        self.config = config or load_lakehouse_config()
        self.bronze_dir = self.config.get_storage_path("bronze")
        self.quarantine_dir = self.config.get_storage_path("quarantine")
        self.db = DuckDBManager(self.config.abs_db_path)

    def _compute_row_hash(self, row: pd.Series) -> str:
        """Compute MD5 hash of core business attributes for change data tracking."""
        val_str = f"{row.get('age')}_{row.get('job')}_{row.get('balance')}_{row.get('day')}_{row.get('month')}"
        return hashlib.md5(val_str.encode("utf-8")).hexdigest()

    def ingest_file(self, source_path: Union[str, Path]) -> Dict[str, Union[int, str]]:
        """
        Ingest raw CSV or Parquet file into Lakehouse Bronze Layer.
        
        Returns:
            Dict containing batch execution metadata.
        """
        src_p = Path(source_path)
        if not src_p.is_absolute():
            # Try resolving from package root or cwd
            if not src_p.exists():
                src_p = Path.cwd() / source_path

        if not src_p.exists():
            raise FileNotFoundError(f"Raw source file not found at: {src_p}")

        # Read source data
        if src_p.suffix.lower() == ".parquet":
            df = pd.read_parquet(src_p)
        else:
            df = pd.read_csv(src_p)

        # Validate with contract
        valid_df, quarantine_df = validate_bronze_data(df)

        batch_id = str(uuid.uuid4())[:8]
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Handle Quarantine if any bad rows
        quarantine_count = 0
        if not quarantine_df.empty and self.config.contracts.allow_quarantine:
            quarantine_count = len(quarantine_df)
            quarantine_df["_batch_id"] = batch_id
            quarantine_df["_quarantined_at"] = now_ts
            quar_file = self.quarantine_dir / f"quarantine_batch_{batch_id}.parquet"
            quarantine_df.to_parquet(quar_file, index=False)

        # Enhance valid records with Bronze Lakehouse Audit Metadata
        valid_df = valid_df.copy()
        if "customer_id" not in valid_df.columns:
            valid_df["customer_id"] = [f"CUST_{batch_id}_{i:06d}" for i in range(len(valid_df))]

        valid_df["_batch_id"] = batch_id
        valid_df["_ingested_at"] = now_ts
        valid_df["_source_file"] = src_p.name
        valid_df["_record_hash"] = valid_df.apply(self._compute_row_hash, axis=1)

        # Save to Bronze Parquet partition
        bronze_file = self.bronze_dir / f"bronze_telemarketing_{batch_id}.parquet"
        valid_df.to_parquet(bronze_file, index=False, compression="snappy")

        # Update DuckDB View over all Bronze Parquets
        bronze_pattern = str(self.bronze_dir / "*.parquet").replace("\\", "/")
        view_sql = f"""
            CREATE OR REPLACE VIEW {self.config.tables.bronze_raw} AS
            SELECT * FROM read_parquet('{bronze_pattern}')
        """
        self.db.execute(view_sql)

        return {
            "batch_id": batch_id,
            "source_file": src_p.name,
            "ingested_records": len(valid_df),
            "quarantined_records": quarantine_count,
            "bronze_file": str(bronze_file),
            "timestamp": now_ts,
        }
