"""
Data Drift and Population Stability Index (PSI) Monitoring Module.
Detects distribution shift between baseline training data and newly incoming inference leads.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DriftMonitor:
    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        self.baseline_df = baseline_df

    @staticmethod
    def calculate_psi(
        expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4
    ) -> float:
        """
        Calculate Population Stability Index (PSI) between expected (training) and actual (production) distribution.
        
        Rules of thumb:
            PSI < 0.10: No significant distribution change.
            0.10 <= PSI < 0.25: Moderate shift detected.
            PSI >= 0.25: Significant drift; Model retraining strongly recommended!
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantile buckets based on expected
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 2:
            return 0.0

        # Bin counts
        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        # Percentages
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Add epsilon to prevent log(0) or division by zero
        expected_pct = np.clip(expected_pct, epsilon, None)
        actual_pct = np.clip(actual_pct, epsilon, None)

        # Re-normalize
        expected_pct /= expected_pct.sum()
        actual_pct /= actual_pct.sum()

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(round(psi_val, 4))

    def evaluate_drift(
        self,
        incoming_df: pd.DataFrame,
        numerical_features: Optional[List[str]] = None,
        psi_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluate drift on numerical features between baseline and incoming dataset.
        """
        if self.baseline_df is None or self.baseline_df.empty:
            raise ValueError("Baseline dataframe is required for drift calculation.")

        num_cols = numerical_features or ["age", "balance", "day", "campaign"]
        common_cols = [c for c in num_cols if c in self.baseline_df.columns and c in incoming_df.columns]

        drift_results = {}
        drift_detected = False

        for col in common_cols:
            exp_vals = pd.to_numeric(self.baseline_df[col], errors="coerce").dropna().values
            act_vals = pd.to_numeric(incoming_df[col], errors="coerce").dropna().values

            if len(exp_vals) == 0 or len(act_vals) == 0:
                continue

            psi_score = self.calculate_psi(exp_vals, act_vals)
            is_drifted = psi_score >= psi_threshold
            if is_drifted:
                drift_detected = True

            status_label = "STABLE"
            if psi_score >= psi_threshold:
                status_label = "SIGNIFICANT_DRIFT"
            elif psi_score >= 0.10:
                status_label = "MODERATE_SHIFT"

            drift_results[col] = {
                "psi": psi_score,
                "status": status_label,
                "is_drifted": is_drifted,
            }

        return {
            "drift_detected": drift_detected,
            "features_evaluated": len(common_cols),
            "psi_threshold": psi_threshold,
            "feature_metrics": drift_results,
            "retrain_recommended": drift_detected,
        }
