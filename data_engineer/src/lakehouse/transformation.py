"""
Transformation Engine for Lakehouse Silver Layer.
Performs data cleaning, conforming, normalization, and splits into Star Schema (Dim / Fact).
"""

import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from ..config import LakehouseConfig, load_lakehouse_config
from .contracts import validate_silver_customer, validate_silver_interaction
from .db import DuckDBManager


class LakehouseTransformation:
    def __init__(self, config: Optional[LakehouseConfig] = None):
        self.config = config or load_lakehouse_config()
        self.silver_dir = self.config.get_storage_path("silver")
        self.db = DuckDBManager(self.config.abs_db_path)

    def process_silver(self) -> Dict[str, int]:
        """
        Process Bronze Parquet tables into Conformed Silver Parquet tables:
          - dim_customer
          - fact_campaign_interaction
        """
        bronze_dir = self.config.get_storage_path("bronze")
        parquet_files = list(bronze_dir.glob("*.parquet"))
        if not parquet_files:
            from .ingestion import LakehouseIngestion
            from ..config import PACKAGE_ROOT
            sample_file = PACKAGE_ROOT / "sample_data" / "bank_raw_sample.csv"
            ing = LakehouseIngestion(self.config)
            ing.ingest_file(sample_file)
            parquet_files = list(bronze_dir.glob("*.parquet"))

        bronze_pattern = str(bronze_dir / "*.parquet").replace("\\", "/")
        df_bronze = self.db.query_df(f"SELECT * FROM read_parquet('{bronze_pattern}')")

        if df_bronze.empty:
            raise ValueError("No records found in Bronze layer to transform!")

        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Clean & Standardize strings
        string_cols = df_bronze.select_dtypes(include=["object"]).columns
        for c in string_cols:
            df_bronze[c] = df_bronze[c].astype(str).str.strip().str.lower()

        # 2. Extract Conformed Customer Dimension
        customer_cols = [
            "customer_id", "age", "job", "marital", "education",
            "default", "balance", "housing", "loan"
        ]
        # Ensure distinct customers (SCD Type 1 latest state)
        df_cust = (
            df_bronze[customer_cols]
            .drop_duplicates(subset=["customer_id"])
            .copy()
        )
        df_cust["age"] = df_cust["age"].astype(int)
        df_cust["balance"] = df_cust["balance"].astype(float)
        df_cust["_processed_at"] = now_ts

        # Validate customer dimension schema
        df_cust = validate_silver_customer(df_cust)

        # 3. Extract Conformed Campaign Interaction Fact
        fact_cols = [
            "customer_id", "contact", "day", "month",
            "duration", "campaign", "pdays", "previous", "poutcome"
        ]
        if "y" in df_bronze.columns:
            fact_cols.append("y")

        df_fact = df_bronze[fact_cols].copy()
        df_fact["interaction_id"] = [f"INT_{i:07d}" for i in range(len(df_fact))]
        df_fact["day"] = df_fact["day"].astype(int)
        df_fact["duration"] = df_fact["duration"].astype(float)
        df_fact["campaign"] = df_fact["campaign"].astype(int)
        df_fact["pdays"] = df_fact["pdays"].astype(int)
        df_fact["previous"] = df_fact["previous"].astype(int)
        df_fact["_processed_at"] = now_ts

        # Validate interaction fact schema
        df_fact = validate_silver_interaction(df_fact)

        # 4. Save to Silver Parquet files
        cust_file = self.silver_dir / f"{self.config.tables.silver_customer}.parquet"
        fact_file = self.silver_dir / f"{self.config.tables.silver_interaction}.parquet"

        df_cust.to_parquet(cust_file, index=False, compression="snappy")
        df_fact.to_parquet(fact_file, index=False, compression="snappy")

        # 5. Register Silver Views in DuckDB
        self.db.register_parquet(self.config.tables.silver_customer, cust_file)
        self.db.register_parquet(self.config.tables.silver_interaction, fact_file)

        return {
            "dim_customer_count": len(df_cust),
            "fact_interaction_count": len(df_fact),
        }
