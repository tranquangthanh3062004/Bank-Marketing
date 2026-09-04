"""
End-to-end pipeline test verifying the complete Data Lakehouse & AI Engine workflow.
"""

from pathlib import Path
import pandas as pd
from data_engineer.src.cli import (
    cmd_batch_score,
    cmd_build_features,
    cmd_ingest,
    cmd_train,
    cmd_transform_gold,
    cmd_transform_silver,
)
from data_engineer.src.config import PACKAGE_ROOT


class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_full_pipeline_e2e():
    sample_file = str(PACKAGE_ROOT / "sample_data" / "bank_raw_sample.csv")
    leads_file = str(PACKAGE_ROOT / "sample_data" / "leads_to_score.csv")
    scored_output = str(PACKAGE_ROOT / "sample_data" / "e2e_scored_leads.csv")

    # 1. Ingest Bronze
    args_ingest = DummyArgs(source=sample_file)
    cmd_ingest(args_ingest)

    # 2. Transform Silver
    cmd_transform_silver(DummyArgs())

    # 3. Transform Gold
    cmd_transform_gold(DummyArgs())

    # 4. Build Feature Store
    cmd_build_features(DummyArgs())

    # 5. Train LightGBM Model
    args_train = DummyArgs(model="lightgbm")
    cmd_train(args_train)

    # 6. Batch Score Leads
    args_score = DummyArgs(input=leads_file, output=scored_output)
    cmd_batch_score(args_score)

    assert Path(scored_output).exists()
    df = pd.read_csv(scored_output)
    assert len(df) == 100
    assert "priority_tier" in df.columns
    assert "predicted_conversion_prob" in df.columns

    # Clean up test output
    Path(scored_output).unlink()
