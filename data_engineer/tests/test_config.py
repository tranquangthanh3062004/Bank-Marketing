"""
Unit tests for configuration loading and path resolution.
"""

from pathlib import Path
from data_engineer.src.config import (
    PACKAGE_ROOT,
    load_lakehouse_config,
    load_model_config,
)


def test_lakehouse_config_loading():
    cfg = load_lakehouse_config()
    assert cfg.name == "bank_marketing_lakehouse"
    assert cfg.contracts.enforce_schema is True
    assert cfg.abs_db_path.name.endswith(".duckdb")
    assert cfg.get_storage_path("bronze").exists()
    assert cfg.get_storage_path("silver").exists()
    assert cfg.get_storage_path("gold").exists()


def test_model_config_loading():
    cfg = load_model_config()
    assert cfg.name == "term_deposit_predictor"
    assert cfg.version == "1.0.0"
    assert "age" in cfg.features.numerical
    assert "job" in cfg.features.categorical
    assert "housing" in cfg.features.binary
    assert cfg.abs_registry_dir.exists()
