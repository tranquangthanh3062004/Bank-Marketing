"""
Lakehouse module.
Provides DuckDB engine, data contracts, Bronze/Silver/Gold transformations and Feature Store.
"""

from .db import DuckDBManager
from .contracts import (
    CustomerLeadPayload,
    validate_bronze_data,
    validate_silver_customer,
    validate_silver_interaction,
)
from .ingestion import LakehouseIngestion
from .transformation import LakehouseTransformation
from .marts import LakehouseMarts
from .feature_store import FeatureStore

__all__ = [
    "DuckDBManager",
    "CustomerLeadPayload",
    "validate_bronze_data",
    "validate_silver_customer",
    "validate_silver_interaction",
    "LakehouseIngestion",
    "LakehouseTransformation",
    "LakehouseMarts",
    "FeatureStore",
]
