"""
Configuration management module.
Loads YAML configurations and resolves absolute paths.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from pydantic import BaseModel, Field

# Determine package root (data_engineer folder)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class StorageConfig(BaseModel):
    base_dir: str
    bronze_dir: str
    silver_dir: str
    gold_dir: str
    quarantine_dir: str
    raw_dir: str

    def get_abs_path(self, rel_path: str) -> Path:
        return PACKAGE_ROOT / rel_path


class TablesConfig(BaseModel):
    bronze_raw: str
    silver_customer: str
    silver_interaction: str
    gold_campaign_mart: str
    gold_segment_mart: str
    feature_store_offline: str
    feature_store_online: str


class ContractsConfig(BaseModel):
    enforce_schema: bool = True
    allow_quarantine: bool = True
    max_quarantine_pct: float = 0.10


class LakehouseConfig(BaseModel):
    name: str
    database_path: str
    storage: StorageConfig
    tables: TablesConfig
    contracts: ContractsConfig

    @property
    def abs_db_path(self) -> Path:
        p = PACKAGE_ROOT / self.database_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_storage_path(self, layer: str) -> Path:
        mapping = {
            "base": self.storage.base_dir,
            "bronze": self.storage.bronze_dir,
            "silver": self.storage.silver_dir,
            "gold": self.storage.gold_dir,
            "quarantine": self.storage.quarantine_dir,
            "raw": self.storage.raw_dir,
        }
        p = PACKAGE_ROOT / mapping[layer]
        p.mkdir(parents=True, exist_ok=True)
        return p


class FeaturesConfig(BaseModel):
    numerical: List[str]
    categorical: List[str]
    binary: List[str]


class ModelConfig(BaseModel):
    name: str
    version: str
    registry_dir: str
    default_model_type: str
    dataset: Dict[str, Any]
    features: FeaturesConfig
    training: Dict[str, Any]
    threshold_tuning: Dict[str, Any]
    serving: Dict[str, Any]

    @property
    def abs_registry_dir(self) -> Path:
        p = PACKAGE_ROOT / self.registry_dir
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_lakehouse_config(config_path: Path = None) -> LakehouseConfig:
    if config_path is None:
        config_path = PACKAGE_ROOT / "config" / "lakehouse_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return LakehouseConfig(**data["lakehouse"])


def load_model_config(config_path: Path = None) -> ModelConfig:
    if config_path is None:
        config_path = PACKAGE_ROOT / "config" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data["model"])
