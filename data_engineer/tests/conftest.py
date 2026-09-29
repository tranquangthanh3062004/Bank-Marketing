"""
Pytest configuration and fixtures for Data Lakehouse & AI Engine tests.
Preserves production lakehouse state by cleaning temporary test batches.
"""

from pathlib import Path
import pytest
from data_engineer.src.config import load_lakehouse_config


@pytest.fixture(scope="session", autouse=True)
def preserve_lakehouse_state():
    """Keep track of original bronze files and clean up newly created ones after tests."""
    cfg = load_lakehouse_config()
    bronze_dir = cfg.get_storage_path("bronze")
    initial_files = set(bronze_dir.glob("*.parquet"))

    yield

    # Clean up test-created bronze files, keeping at least 1 valid file
    current_files = set(bronze_dir.glob("*.parquet"))
    new_files = current_files - initial_files
    for f in new_files:
        try:
            f.unlink()
        except Exception:
            pass
