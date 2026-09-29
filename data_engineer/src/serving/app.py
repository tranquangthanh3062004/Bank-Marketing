"""
FastAPI REST Service for Real-Time Telemarketing Lead Scoring and Explainable AI.
Supports Dual-Model Architecture: Pre-Call Prioritization & Post-Call Evaluation.
Enterprise Features:
- Closed-Loop CTI / Call Center Feedback Integration
- Prometheus Metrics (/metrics) for APM and Grafana Dashboards
- Population Stability Index (PSI) Drift Detection
- API Key Security and Healthchecks
"""

from contextlib import asynccontextmanager
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import PlainTextResponse
import numpy as np
import pandas as pd
from ..ai_engine.drift_monitor import DriftMonitor
from ..ai_engine.explainability import ModelExplainer
from ..ai_engine.registry import ModelRegistry
from ..config import load_lakehouse_config, load_model_config
from ..lakehouse.contracts import CustomerLeadPayload
from ..lakehouse.feature_store import FeatureStore
from .feedback_loop import CallFeedbackPayload, FeedbackLoopManager
from .metrics import metrics


# Context state storage
state: Dict[str, Any] = {
    "model": None,
    "pipeline": None,
    "metadata": None,
    "explainer": None,
    "pre_call_model": None,
    "pre_call_pipeline": None,
    "pre_call_metadata": None,
    "pre_call_explainer": None,
    "feature_store": None,
    "model_registry": None,
}

DEFAULT_API_KEY = "bm-enterprise-secret-key-2026"


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Validate X-API-KEY header if REQUIRE_API_KEY is enabled."""
    required = os.environ.get("REQUIRE_API_KEY", "false").lower() == "true"
    expected_key = os.environ.get("BANK_MARKETING_API_KEY", DEFAULT_API_KEY)
    if required:
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header",
            )
    return x_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    model_cfg = load_model_config()
    lake_cfg = load_lakehouse_config()
    registry = ModelRegistry(model_cfg)
    state["model_registry"] = registry
    state["feature_store"] = FeatureStore(lake_cfg)

    # Load post_call model (default best model)
    if registry.is_model_available(mode="post_call"):
        model, pipeline, metadata = registry.load_active_model(mode="post_call")
        state["model"] = model
        state["pipeline"] = pipeline
        state["metadata"] = metadata
        state["explainer"] = ModelExplainer(model, pipeline, metadata.get("model_type", "lightgbm"))

    # Load pre_call model
    if registry.is_model_available(mode="pre_call"):
        p_model, p_pipeline, p_metadata = registry.load_active_model(mode="pre_call")
        state["pre_call_model"] = p_model
        state["pre_call_pipeline"] = p_pipeline
        state["pre_call_metadata"] = p_metadata
        state["pre_call_explainer"] = ModelExplainer(p_model, p_pipeline, p_metadata.get("model_type", "lightgbm"))
    elif state.get("model"):
        # Fallback to default if pre-call model is not trained separately
        state["pre_call_model"] = state["model"]
        state["pre_call_pipeline"] = state["pipeline"]
        state["pre_call_metadata"] = state["metadata"]
        state["pre_call_explainer"] = state["explainer"]

    yield
    # Shutdown logic
    state.clear()


app = FastAPI(
    title="Bank Marketing Enterprise Lead Scoring & AI Lakehouse API",
    description="Enterprise Real-Time Inference, Explainability, and Closed-Loop CRM Integration",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root():
    metrics.record_request()
    return {
        "service": "Bank Marketing Enterprise AI Lead Scoring Engine",
        "status": "online",
        "docs_url": "/docs",
        "metrics_url": "/metrics",
        "version": "2.0.0",
        "supported_modes": ["pre_call", "post_call"],
        "compliance": ["Decree 13/2023/ND-CP", "SR 11-7 Ethical AI", "OpenMetrics/Prometheus"],
    }


@app.get("/health", tags=["General"])
def health():
    metrics.record_request()
    post_ready = state.get("model") is not None
    pre_ready = state.get("pre_call_model") is not None
    return {
        "status": "healthy" if (post_ready or pre_ready) else "uninitialized",
        "model_loaded": post_ready or pre_ready,
        "pre_call_model_loaded": pre_ready,
        "post_call_model_loaded": post_ready,
        "pre_call_threshold": state["pre_call_metadata"].get("optimal_threshold") if pre_ready else None,
        "post_call_threshold": state["metadata"].get("optimal_threshold") if post_ready else None,
    }


@app.get("/metrics", tags=["Observability"], response_class=PlainTextResponse)
def get_prometheus_metrics():
    """Exposes real-time Prometheus / Grafana metrics endpoint."""
    return metrics.to_prometheus_format()


@app.get("/api/v1/model/info", tags=["AI Engine"])
def get_model_info(mode: str = "post_call"):
    metrics.record_request()
    if mode == "pre_call" and state.get("pre_call_metadata"):
        return state["pre_call_metadata"]
    if not state.get("model"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    return state["metadata"]


def _score_lead_internal(payload: CustomerLeadPayload, mode: str = "post_call") -> Dict[str, Any]:
    t0 = time.time()
    target_model = state.get("pre_call_model") if mode == "pre_call" else state.get("model")
    target_pipeline = state.get("pre_call_pipeline") if mode == "pre_call" else state.get("pipeline")
    target_explainer = state.get("pre_call_explainer") if mode == "pre_call" else state.get("explainer")
    target_meta = state.get("pre_call_metadata") if mode == "pre_call" else state.get("metadata")

    if not target_model or not target_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model for mode '{mode}' is not loaded. Please train a model first.",
        )

    lead_dict = payload.model_dump()
    if mode == "pre_call":
        lead_dict["duration"] = 0

    derived_dict = FeatureStore.derive_lead_features(lead_dict)
    df_lead = pd.DataFrame([derived_dict])

    opt_threshold = target_meta.get("optimal_threshold", 0.5) if target_meta else 0.5
    X_trans = target_pipeline.transform(df_lead)
    prob = float(target_model.predict_proba(X_trans)[0, 1])
    is_recommended = bool(prob >= opt_threshold)

    # Assign Priority Tier
    if prob >= 0.70:
        tier = "Tier 1 (Hot)"
        action = "Chuyển ngay cho Senior Telesales; gọi trong 30 phút"
    elif prob >= 0.45:
        tier = "Tier 2 (Warm)"
        action = "Gán cho Telesales tiêu chuẩn; ưu tiên gọi trong ngày"
    elif prob >= 0.25:
        tier = "Tier 3 (Neutral)"
        action = "Gửi tin nhắn SMS / Email giới thiệu trước"
    else:
        tier = "Tier 4 (Cold)"
        action = "Không gọi điện trong đợt này"

    drivers = {"positive_drivers": [], "negative_drivers": []}
    if target_explainer:
        try:
            drivers = target_explainer.explain_single_lead(X_trans[0], top_k=3)
        except Exception:
            pass

    dur_sec = time.time() - t0
    metrics.record_prediction(tier, mode, dur_sec)

    return {
        "customer_id": payload.customer_id,
        "mode": mode,
        "conversion_probability": round(prob, 4),
        "is_recommended_call": is_recommended,
        "decision_threshold": round(opt_threshold, 3),
        "priority_tier": tier,
        "recommended_action": action,
        "top_positive_drivers": drivers.get("positive_drivers", []),
        "top_negative_barriers": drivers.get("negative_drivers", []),
        "latency_ms": round(dur_sec * 1000, 2),
    }


@app.post("/api/v1/predict/pre-call", tags=["Inference"])
def predict_pre_call(
    payload: CustomerLeadPayload,
    _auth: Optional[str] = Depends(verify_api_key),
):
    """Score incoming lead BEFORE dialing (duration excluded to prevent lookahead bias)."""
    return _score_lead_internal(payload, mode="pre_call")


@app.post("/api/v1/predict/post-call", tags=["Inference"])
def predict_post_call(
    payload: CustomerLeadPayload,
    _auth: Optional[str] = Depends(verify_api_key),
):
    """Score customer AFTER conversation (includes actual call duration)."""
    return _score_lead_internal(payload, mode="post_call")


@app.post("/api/v1/predict", tags=["Inference"])
def predict_lead(
    payload: CustomerLeadPayload,
    _auth: Optional[str] = Depends(verify_api_key),
):
    """Smart router: routes to pre-call if duration <= 0, else post-call."""
    mode = "pre_call" if (payload.duration is None or payload.duration == 0) else "post_call"
    return _score_lead_internal(payload, mode=mode)


@app.post("/api/v1/batch-predict", tags=["Inference"])
def batch_predict(
    leads: List[CustomerLeadPayload],
    mode: str = "auto",
    _auth: Optional[str] = Depends(verify_api_key),
):
    """Batch score a list of customer leads."""
    results = []
    for lead in leads:
        lead_mode = mode
        if mode == "auto":
            lead_mode = "pre_call" if (lead.duration is None or lead.duration == 0) else "post_call"
        results.append(_score_lead_internal(lead, mode=lead_mode))
    return {"total_scored": len(results), "leads": results}


@app.post("/api/v1/telemarketing/feedback", tags=["Closed-Loop CRM"])
def submit_call_feedback(
    payload: CallFeedbackPayload,
    _auth: Optional[str] = Depends(verify_api_key),
):
    """
    Record Telesales call outcome (agreed/declined/callback/unreachable)
    and append to the Lakehouse feedback log for closed-loop continuous learning.
    """
    loop_mgr = FeedbackLoopManager()
    res = loop_mgr.record_feedback(payload)
    metrics.record_feedback(payload.outcome)
    return res


@app.post("/api/v1/monitor/drift", tags=["MLOps"])
def check_population_drift(
    leads: List[CustomerLeadPayload],
    _auth: Optional[str] = Depends(verify_api_key),
):
    """
    Evaluate Population Stability Index (PSI) drift between baseline and incoming leads batch.
    """
    fs: FeatureStore = state.get("feature_store")
    if not fs:
        raise HTTPException(status_code=503, detail="Feature store not initialized.")
    df_baseline = fs.get_offline_features()
    df_incoming = pd.DataFrame([l.model_dump() for l in leads])
    monitor = DriftMonitor(baseline_df=df_baseline)
    res = monitor.evaluate_drift(df_incoming)
    if res["drift_detected"]:
        metrics.record_drift_event()
    return res


@app.get("/api/v1/features/{customer_id}", tags=["Feature Store"])
def get_customer_features(
    customer_id: str,
    _auth: Optional[str] = Depends(verify_api_key),
):
    """Retrieve online features for an existing customer from the Lakehouse."""
    fs: FeatureStore = state.get("feature_store")
    if not fs:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Feature store uninitialized.")

    feat = fs.get_online_features(customer_id)
    if not feat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found in Lakehouse.",
        )
    return feat
