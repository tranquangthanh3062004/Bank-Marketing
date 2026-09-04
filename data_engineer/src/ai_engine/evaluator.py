"""
Model Evaluator and Threshold Optimizer.
Measures discrimination metrics and tunes optimal operational decision threshold.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from ..config import ModelConfig, load_model_config


class ModelEvaluator:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()

    def tune_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Find decision threshold that maximizes F1-Score or Expected Business Profit.
        """
        deposit_profit = self.config.threshold_tuning.get("deposit_profit_eur", 150.0)
        call_cost = self.config.threshold_tuning.get("call_cost_eur", 5.0)

        thresholds = np.linspace(0.05, 0.95, 91)
        best_f1 = -1.0
        best_profit = -float("inf")
        opt_threshold_f1 = 0.5
        opt_threshold_profit = 0.5

        records = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            total_calls = tp + fp
            total_conversions = tp
            net_profit = (total_conversions * deposit_profit) - (total_calls * call_cost)

            if f1 > best_f1:
                best_f1 = f1
                opt_threshold_f1 = float(round(t, 3))

            if net_profit > best_profit:
                best_profit = net_profit
                opt_threshold_profit = float(round(t, 3))

            records.append({
                "threshold": round(float(t), 3),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "calls": int(total_calls),
                "conversions": int(total_conversions),
                "net_profit_eur": round(float(net_profit), 2),
            })

        chosen_threshold = opt_threshold_f1
        if self.config.threshold_tuning.get("metric") == "net_profit":
            chosen_threshold = opt_threshold_profit

        return {
            "optimal_threshold": chosen_threshold,
            "optimal_threshold_f1": opt_threshold_f1,
            "max_f1": round(best_f1, 4),
            "optimal_threshold_profit": opt_threshold_profit,
            "max_profit_eur": round(best_profit, 2),
            "curve_samples": records[::5],  # sampled for logging
        }

    def evaluate(
        self,
        model: Any,
        X_trans: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Compute full suite of classification and business metrics."""
        y_proba = model.predict_proba(X_trans)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        return {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "threshold_used": round(threshold, 3),
            "confusion_matrix": cm,
        }
