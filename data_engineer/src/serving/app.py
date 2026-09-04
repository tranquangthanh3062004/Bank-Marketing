"""
FastAPI REST Service for Real-Time Telemarketing Lead Scoring and Explainable AI.
Provides sub-second inference with SHAP factor breakdown.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
import numpy as np
import pandas as pd
from ..ai_engine.explainability import ModelExplainer
from ..ai_engine.registry import ModelRegistry
from ..config import load_lakehouse_config, load_model_config
from ..lakehouse.contracts import CustomerLeadPayload
from ..lakehouse.feature_store import FeatureStore


# Context state storage
state: Dict[str, Any] = {
    "model": None,
    "pipeline": None,
    "metadata": None,
    "explainer": None,
    "feature_store": None,
    "model_registry": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    model_cfg = load_model_config()
    lake_cfg = load_lakehouse_config()
    registry = ModelRegistry(model_cfg)
    state["model_registry"] = registry
    state["feature_store"] = FeatureStore(lake_cfg)

    if registry.is_model_available():
        model, pipeline, metadata = registry.load_active_model()
        state["model"] = model
        state["pipeline"] = pipeline
        state["metadata"] = metadata

        explainer = ModelExplainer(model, pipeline, metadata.get("model_type", "lightgbm"))
        state["explainer"] = explainer
    yield
    # Shutdown logic
    state.clear()


app = FastAPI(
    title="Bank Marketing AI Lakehouse Lead Scoring API",
    description="Enterprise Real-Time Inference & Explainability Service for Term Deposit Conversion",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Bank Marketing AI Lead Scoring Engine",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", tags=["General"])
def health():
    is_ready = state.get("model") is not None
    return {
        "status": "healthy" if is_ready else "uninitialized",
        "model_loaded": is_ready,
        "model_version": state["metadata"].get("version") if is_ready else None,
        "model_type": state["metadata"].get("model_type") if is_ready else None,
    }


@app.get("/api/v1/model/info", tags=["AI Engine"])
def get_model_info():
    if not state.get("model"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
    return state["metadata"]


@app.post("/api/v1/predict", tags=["Inference"])
def predict_lead(payload: CustomerLeadPayload):
    """
    Predict conversion probability and explain key drivers for an incoming lead.
    """
    if not state.get("model"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train a model first.",
        )

    # Convert payload to dict and derive feature store columns
    lead_dict = payload.model_dump()
    derived_dict = FeatureStore.derive_lead_features(lead_dict)
    df_lead = pd.DataFrame([derived_dict])

    pipeline = state["pipeline"]
    model = state["model"]
    explainer = state["explainer"]
    metadata = state["metadata"]
    opt_threshold = metadata.get("optimal_threshold", 0.5)

    # Preprocessing
    X_trans = pipeline.transform(df_lead)

    # Inference
    prob = float(model.predict_proba(X_trans)[0, 1])
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

    # SHAP local attribution
    drivers = {"positive_drivers": [], "negative_drivers": []}
    if explainer:
        try:
            drivers = explainer.explain_single_lead(X_trans[0], top_k=3)
        except Exception:
            pass

    return {
        "customer_id": payload.customer_id,
        "conversion_probability": round(prob, 4),
        "is_recommended_call": is_recommended,
        "decision_threshold": round(opt_threshold, 3),
        "priority_tier": tier,
        "recommended_action": action,
        "top_positive_drivers": drivers["positive_drivers"],
        "top_negative_barriers": drivers["negative_drivers"],
    }


@app.post("/api/v1/batch-predict", tags=["Inference"])
def batch_predict(leads: List[CustomerLeadPayload]):
    """
    Batch score a list of customer leads.
    """
    if not state.get("model"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")

    results = []
    for lead in leads:
        results.append(predict_lead(lead))
    return {"total_scored": len(results), "leads": results}


@app.get("/api/v1/features/{customer_id}", tags=["Feature Store"])
def get_customer_features(customer_id: str):
    """
    Retrieve online features for an existing customer from the Lakehouse.
    """
    fs: FeatureStore = state.get("feature_store")
    if not fs:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Feature store uninitialized.")

    feat = fs.get_online_features(customer_id)
    if not feat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found in Lakehouse.")
    return feat
