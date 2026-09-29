"""
Enterprise Unit and Integration Tests.
Verifies PII Masking, SR 11-7 AI Fairness Audit, Prometheus Metrics,
Closed-Loop CTI Feedback, and Enterprise DAG Workflow Orchestration.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient
from data_engineer.src.ai_engine.fairness_audit import FairnessAuditor
from data_engineer.src.governance.pii_masking import PIISecurityManager
from data_engineer.src.orchestration.workflow_runner import EnterpriseWorkflowOrchestrator
from data_engineer.src.serving.app import app
from data_engineer.src.serving.feedback_loop import CallFeedbackPayload, FeedbackLoopManager
from data_engineer.src.serving.metrics import metrics


def test_pii_security_manager():
    sec = PIISecurityManager(salt="test_salt_123")

    # Tokenization
    t1 = sec.tokenize_customer_id("CUST_00123")
    t2 = sec.tokenize_customer_id("CUST_00123")
    t3 = sec.tokenize_customer_id("CUST_00456")

    assert t1.startswith("TOKEN_")
    assert t1 == t2  # Deterministic for analytical joins
    assert t1 != t3  # Distinct per customer

    # Phone masking
    masked = sec.mask_phone_number("0912345678")
    assert masked == "091****678"

    # De-identification of dataframe
    df = pd.DataFrame({
        "customer_id": ["CUST_01", "CUST_02"],
        "balance": [1250, 4800],
    })
    df_clean = sec.deidentify_dataframe(df, mask_financials=True)
    assert not df_clean["customer_id"].str.startswith("CUST").any()
    assert df_clean["balance"].iloc[0] == 1000
    assert df_clean["balance"].iloc[1] == 4500


def test_fairness_auditor_sr_11_7():
    auditor = FairnessAuditor()

    # Synthetic predictions and protected attribute
    n = 200
    np.random.seed(42)
    ages = np.random.randint(20, 80, size=n)
    X_df = pd.DataFrame({"age": ages})
    y_true = (np.random.rand(n) > 0.6).astype(int)
    y_pred = (np.random.rand(n) > 0.5).astype(int)

    report = auditor.audit_model(X_df, y_true, y_pred, protected_attribute="age")
    assert "metrics" in report
    assert "disparate_impact_ratio" in report["metrics"]
    assert "equal_opportunity_difference" in report["metrics"]
    assert "sr_11_7_status" in report
    assert report["sample_counts"]["protected_count"] > 0


def test_prometheus_metrics_endpoint():
    with TestClient(app) as client:
        # Trigger request
        res = client.get("/health")
        assert res.status_code == 200

        # Scrape metrics
        res_metrics = client.get("/metrics")
        assert res_metrics.status_code == 200
        text = res_metrics.text
        assert "bank_marketing_http_requests_total" in text
        assert "bank_marketing_predictions_total" in text
        assert "bank_marketing_latency_seconds_bucket" in text


def test_closed_loop_feedback_endpoint():
    with TestClient(app) as client:
        payload = {
            "customer_id": "CUST_TEST_007",
            "agent_id": "AGENT_99",
            "call_duration": 210,
            "outcome": "agreed",
            "notes": "Customer agreed to 12-month term deposit",
        }
        res = client.post("/api/v1/telemarketing/feedback", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "RECORDED"
        assert data["customer_id"] == "CUST_TEST_007"
        assert data["mapped_y"] == "yes"


def test_enterprise_workflow_orchestrator():
    orchestrator = EnterpriseWorkflowOrchestrator()
    res = orchestrator.run_daily_pipeline()

    assert res["status"] == "SUCCESS"
    assert res["ingested_records"] > 0
    assert "pre_call" in res["models"]
    assert "post_call" in res["models"]
    assert "fairness_audit" in res["models"]["pre_call"]
