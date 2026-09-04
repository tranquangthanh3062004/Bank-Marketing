"""
Unit and Integration tests for Lakehouse Medallion layers (Bronze, Silver, Gold).
"""

from pathlib import Path
import pandas as pd
import pytest
from data_engineer.src.config import PACKAGE_ROOT, load_lakehouse_config
from data_engineer.src.lakehouse.db import DuckDBManager
from data_engineer.src.lakehouse.ingestion import LakehouseIngestion
from data_engineer.src.lakehouse.marts import LakehouseMarts
from data_engineer.src.lakehouse.transformation import LakehouseTransformation


def test_lakehouse_ingestion_and_bronze():
    cfg = load_lakehouse_config()
    sample_file = PACKAGE_ROOT / "sample_data" / "bank_raw_sample.csv"
    assert sample_file.exists()

    ingestion = LakehouseIngestion(cfg)
    res = ingestion.ingest_file(sample_file)

    assert res["ingested_records"] > 0
    assert "batch_id" in res
    assert Path(res["bronze_file"]).exists()

    # Query DuckDB Bronze view
    db = DuckDBManager(cfg.abs_db_path)
    df_bronze = db.query_df(f"SELECT COUNT(*) as cnt FROM {cfg.tables.bronze_raw}")
    assert df_bronze.iloc[0]["cnt"] >= res["ingested_records"]


def test_lakehouse_transformation_silver():
    cfg = load_lakehouse_config()
    trans = LakehouseTransformation(cfg)
    res = trans.process_silver()

    assert res["dim_customer_count"] > 0
    assert res["fact_interaction_count"] > 0

    # Verify silver parquet files exist
    cust_file = cfg.get_storage_path("silver") / f"{cfg.tables.silver_customer}.parquet"
    fact_file = cfg.get_storage_path("silver") / f"{cfg.tables.silver_interaction}.parquet"
    assert cust_file.exists()
    assert fact_file.exists()

    # Verify DuckDB Silver Views
    db = DuckDBManager(cfg.abs_db_path)
    df_cust = db.query_df(f"SELECT * FROM {cfg.tables.silver_customer} LIMIT 5")
    assert not df_cust.empty
    assert "balance" in df_cust.columns


def test_lakehouse_gold_marts():
    cfg = load_lakehouse_config()
    marts = LakehouseMarts(cfg)
    res = marts.build_marts()

    assert res["campaign_mart_rows"] > 0
    assert res["segment_mart_rows"] > 0

    # Query DuckDB Gold Views
    db = DuckDBManager(cfg.abs_db_path)
    df_camp = db.query_df(f"SELECT * FROM {cfg.tables.gold_campaign_mart} LIMIT 5")
    assert not df_camp.empty
    assert "conversion_rate" in df_camp.columns
