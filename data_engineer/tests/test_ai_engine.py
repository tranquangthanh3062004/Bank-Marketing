"""
Unit and Integration tests for AI Engine (Pipeline, Trainer, Evaluator, SHAP Explainer, Registry).
"""

import numpy as np
import pandas as pd
import pytest
from data_engineer.src.ai_engine.evaluator import ModelEvaluator
from data_engineer.src.ai_engine.explainability import ModelExplainer
from data_engineer.src.ai_engine.features import FeaturePipeline
from data_engineer.src.ai_engine.registry import ModelRegistry
from data_engineer.src.ai_engine.trainer import ModelTrainer
from data_engineer.src.config import load_model_config
from data_engineer.src.lakehouse.feature_store import FeatureStore


def test_feature_pipeline_fit_transform():
    fs = FeatureStore()
    df_feat = fs.get_offline_features()
    assert not df_feat.empty

    pipeline = FeaturePipeline()
    X = df_feat.drop(columns=["y"])
    X_trans = pipeline.fit_transform(X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(df_feat)
    assert len(pipeline.feature_names) == X_trans.shape[1]


def test_model_training_lightgbm():
    fs = FeatureStore()
    df_feat = fs.get_offline_features()

    trainer = ModelTrainer()
    train_res = trainer.train(df_feat, model_type="lightgbm")

    model = train_res["model"]
    pipeline = train_res["pipeline"]
    splits = train_res["data_splits"]

    assert model is not None
    assert splits["X_train_trans"].shape[0] > 0
    assert splits["X_val_trans"].shape[0] > 0
    assert splits["X_test_trans"].shape[0] > 0


def test_model_evaluation_and_threshold_tuning():
    fs = FeatureStore()
    df_feat = fs.get_offline_features()

    trainer = ModelTrainer()
    train_res = trainer.train(df_feat, model_type="lightgbm")
    model = train_res["model"]
    splits = train_res["data_splits"]

    evaluator = ModelEvaluator()
    y_val_proba = model.predict_proba(splits["X_val_trans"])[:, 1]

    threshold_info = evaluator.tune_threshold(splits["y_val"], y_val_proba)
    assert "optimal_threshold" in threshold_info
    assert 0.05 <= threshold_info["optimal_threshold"] <= 0.95

    metrics = evaluator.evaluate(
        model,
        splits["X_test_trans"],
        splits["y_test"],
        threshold=threshold_info["optimal_threshold"],
    )
    assert "roc_auc" in metrics
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert "f1_score" in metrics


def test_shap_explainability():
    fs = FeatureStore()
    df_feat = fs.get_offline_features()

    trainer = ModelTrainer()
    train_res = trainer.train(df_feat, model_type="lightgbm")
    model = train_res["model"]
    pipeline = train_res["pipeline"]
    splits = train_res["data_splits"]

    explainer = ModelExplainer(model, pipeline, model_type="lightgbm")
    importance = explainer.get_global_importance(splits["X_train_trans"], top_n=5)

    assert len(importance) > 0
    assert "feature" in importance[0]
    assert "importance" in importance[0]

    # Test single lead explanation
    sample_lead = splits["X_test_trans"][0]
    explanation = explainer.explain_single_lead(sample_lead, top_k=3)
    assert "positive_drivers" in explanation
    assert "negative_drivers" in explanation


def test_model_registry_save_and_load():
    fs = FeatureStore()
    df_feat = fs.get_offline_features()

    trainer = ModelTrainer()
    train_res = trainer.train(df_feat, model_type="lightgbm")
    model = train_res["model"]
    pipeline = train_res["pipeline"]
    splits = train_res["data_splits"]

    evaluator = ModelEvaluator()
    y_val_proba = model.predict_proba(splits["X_val_trans"])[:, 1]
    threshold_info = evaluator.tune_threshold(splits["y_val"], y_val_proba)
    test_metrics = evaluator.evaluate(model, splits["X_test_trans"], splits["y_test"])

    registry = ModelRegistry()
    res = registry.save_model(
        model=model,
        pipeline=pipeline,
        metrics=test_metrics,
        threshold_info=threshold_info,
        model_type="lightgbm",
    )
    assert registry.is_model_available()

    loaded_model, loaded_pipeline, metadata = registry.load_active_model()
    assert loaded_model is not None
    assert metadata["model_type"] == "lightgbm"


def test_drift_monitor():
    from data_engineer.src.ai_engine.drift_monitor import DriftMonitor

    baseline_data = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "balance": [100, 500, 1000, 1500, 2000, 2500, 3000, 3500],
        "campaign": [1, 2, 1, 3, 2, 1, 2, 1],
    })

    # Test stable incoming
    stable_incoming = baseline_data.copy()
    monitor = DriftMonitor(baseline_df=baseline_data)
    res_stable = monitor.evaluate_drift(stable_incoming)
    assert res_stable["drift_detected"] is False
    assert res_stable["retrain_recommended"] is False

    # Test drifted incoming
    drifted_incoming = pd.DataFrame({
        "age": [80, 85, 90, 82, 88, 89, 92, 85],
        "balance": [50000, 60000, 75000, 80000, 55000, 70000, 65000, 90000],
        "campaign": [10, 12, 15, 11, 14, 16, 12, 13],
    })
    res_drift = monitor.evaluate_drift(drifted_incoming)
    assert res_drift["drift_detected"] is True
    assert res_drift["retrain_recommended"] is True


def test_dual_mode_training():
    fs = FeatureStore()
    df_feat = fs.get_offline_features()

    trainer = ModelTrainer()
    # Test pre-call model (no duration in features)
    res_pre = trainer.train(df_feat, model_type="lightgbm", mode="pre_call")
    assert "duration" not in res_pre["pipeline"].numerical_cols

    # Test post-call model (duration included)
    res_post = trainer.train(df_feat, model_type="lightgbm", mode="post_call")
    assert "duration" in res_post["pipeline"].numerical_cols
