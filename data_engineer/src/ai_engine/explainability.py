"""
SHAP Explainability and Model Interpretability Engine.
Provides Global Feature Importance and Local Instance-Level Attribution for Call Agents.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap
from .features import FeaturePipeline


class ModelExplainer:
    def __init__(self, model: Any, pipeline: FeaturePipeline, model_type: str = "lightgbm"):
        self.model = model
        self.pipeline = pipeline
        self.model_type = model_type
        self.feature_names = pipeline.feature_names
        self.explainer = None

    def initialize_explainer(self, background_data: np.ndarray) -> None:
        """Initialize SHAP explainer."""
        if self.model_type == "lightgbm":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # For linear models or generic models, use LinearExplainer with sample background
            sample_bg = background_data[:min(100, len(background_data))]
            self.explainer = shap.LinearExplainer(self.model, sample_bg)

    def get_global_importance(self, X_trans: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
        """Calculate mean absolute SHAP values for global ranking."""
        if self.explainer is None:
            self.initialize_explainer(X_trans)

        shap_vals = self.explainer.shap_values(X_trans[:min(300, len(X_trans))])
        if isinstance(shap_vals, list):
            # For binary classification some versions return [neg_vals, pos_vals]
            shap_matrix = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            shap_matrix = shap_vals

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

        importance_list = []
        for name, score in zip(self.feature_names, mean_abs_shap):
            importance_list.append({
                "feature": name,
                "importance": round(float(score), 4),
            })

        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        return importance_list[:top_n]

    def explain_single_lead(
        self, lead_features_trans: np.ndarray, top_k: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Explain single prospect lead for telemarketing agents.
        Returns top drivers that increase probability (positive) and decrease it (negative).
        """
        if self.explainer is None:
            self.initialize_explainer(lead_features_trans)

        lead_vector = lead_features_trans.reshape(1, -1)
        shap_vals = self.explainer.shap_values(lead_vector)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals

        contributions = []
        for name, val in zip(self.feature_names, vals):
            contributions.append({
                "feature": name,
                "impact": round(float(val), 4),
            })

        positive_drivers = sorted(
            [c for c in contributions if c["impact"] > 0],
            key=lambda x: x["impact"],
            reverse=True
        )[:top_k]

        negative_drivers = sorted(
            [c for c in contributions if c["impact"] < 0],
            key=lambda x: x["impact"]
        )[:top_k]

        return {
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
        }
