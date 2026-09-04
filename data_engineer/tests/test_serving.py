"""
Integration tests for Serving (Batch Lead Scorer & FastAPI REST Endpoints).
"""

from pathlib import Path
import pandas as pd
from starlette.testclient import TestClient
from data_engineer.src.config import PACKAGE_ROOT
from data_engineer.src.serving.app import app
from data_engineer.src.serving.batch_scorer import BatchLeadScorer


def test_batch_lead_scorer():
    leads_file = PACKAGE_ROOT / "sample_data" / "leads_to_score.csv"
    output_file = PACKAGE_ROOT / "sample_data" / "test_scored_output.csv"

    scorer = BatchLeadScorer()
    df_scored = scorer.score_file(leads_file, output_file)

    assert not df_scored.empty
    assert "predicted_conversion_prob" in df_scored.columns
    assert "priority_tier" in df_scored.columns
    assert "telesales_action" in df_scored.columns
    assert output_file.exists()

    # Clean up test output
    if output_file.exists():
        output_file.unlink()


def test_fastapi_endpoints():
    with TestClient(app) as client:
        # Health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        data_health = res_health.json()
        assert data_health["status"] == "healthy"
        assert data_health["model_loaded"] is True

        # Model Info
        res_info = client.get("/api/v1/model/info")
        assert res_info.status_code == 200
        data_info = res_info.json()
        assert "optimal_threshold" in data_info

        # Single Lead Predict
        payload = {
            "customer_id": "TEST_LEAD_01",
            "age": 42,
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 2500.0,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 18,
            "month": "aug",
            "duration": 320,
            "campaign": 1,
            "pdays": 120,
            "previous": 2,
            "poutcome": "success",
        }
        res_pred = client.post("/api/v1/predict", json=payload)
        assert res_pred.status_code == 200
        data_pred = res_pred.json()

        assert "conversion_probability" in data_pred
        assert 0.0 <= data_pred["conversion_probability"] <= 1.0
        assert "priority_tier" in data_pred
        assert "recommended_action" in data_pred
        assert "top_positive_drivers" in data_pred
        assert "top_negative_barriers" in data_pred

        # Batch Predict
        res_batch = client.post("/api/v1/batch-predict", json=[payload, payload])
        assert res_batch.status_code == 200
        data_batch = res_batch.json()
        assert data_batch["total_scored"] == 2
