"""
Model Risk Management (MRM) and AI Fairness Auditing Module.
Complies with Federal Reserve SR 11-7 standards and Ethical AI guidelines.
Measures Disparate Impact Ratio (DIR), Equal Opportunity Difference (EOD),
and Demographic Parity across protected groups.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class FairnessAuditor:
    def __init__(self, privileged_group: str = "reference", unprivileged_group: str = "protected"):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    @staticmethod
    def disparate_impact_ratio(
        y_pred: np.ndarray, protected_mask: np.ndarray
    ) -> float:
        """
        Disparate Impact Ratio (DIR) = P(Y_hat=1 | Protected) / P(Y_hat=1 | Privileged)
        Standard rule: 0.80 <= DIR <= 1.25 (Four-fifths / 80% rule).
        """
        protected_favorable = np.mean(y_pred[protected_mask == 1]) if np.sum(protected_mask == 1) > 0 else 0.0
        privileged_favorable = np.mean(y_pred[protected_mask == 0]) if np.sum(protected_mask == 0) > 0 else 0.0

        if privileged_favorable == 0:
            return 1.0
        return float(round(protected_favorable / privileged_favorable, 4))

    @staticmethod
    def equal_opportunity_difference(
        y_true: np.ndarray, y_pred: np.ndarray, protected_mask: np.ndarray
    ) -> float:
        """
        Equal Opportunity Difference = TPR(Protected) - TPR(Privileged)
        Ideal: 0.0 (Acceptable threshold: |EOD| <= 0.10).
        """
        # True Positive Rate for protected
        p_actual_pos = (protected_mask == 1) & (y_true == 1)
        tpr_protected = (
            np.mean(y_pred[p_actual_pos] == 1) if np.sum(p_actual_pos) > 0 else 0.0
        )

        # True Positive Rate for privileged
        u_actual_pos = (protected_mask == 0) & (y_true == 1)
        tpr_privileged = (
            np.mean(y_pred[u_actual_pos] == 1) if np.sum(u_actual_pos) > 0 else 0.0
        )

        return float(round(tpr_protected - tpr_privileged, 4))

    def audit_model(
        self,
        X_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: str = "age",
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive SR 11-7 fairness audit.
        """
        if protected_attribute not in X_df.columns:
            # Fallback if attribute is missing
            return {
                "status": "SKIPPED",
                "reason": f"Protected attribute '{protected_attribute}' not in dataset",
            }

        if protected_attribute == "age":
            # Senior (>= 60) vs Non-Senior (< 60)
            protected_mask = (X_df["age"] >= 60).astype(int).values
            group_labels = {"protected": "Senior (>=60)", "privileged": "Non-Senior (<60)"}
        elif protected_attribute == "marital":
            # Single vs Married
            protected_mask = (X_df["marital"].astype(str).str.lower() == "single").astype(int).values
            group_labels = {"protected": "Single", "privileged": "Married/Other"}
        else:
            protected_mask = (X_df[protected_attribute] == 1).astype(int).values
            group_labels = {"protected": "Protected", "privileged": "Privileged"}

        dir_score = self.disparate_impact_ratio(y_pred, protected_mask)
        eod_score = self.equal_opportunity_difference(y_true, y_pred, protected_mask)

        # Compliance evaluation
        is_dir_compliant = 0.80 <= dir_score <= 1.25
        is_eod_compliant = abs(eod_score) <= 0.10
        overall_compliant = is_dir_compliant and is_eod_compliant

        return {
            "attribute": protected_attribute,
            "groups": group_labels,
            "sample_counts": {
                "protected_count": int(np.sum(protected_mask == 1)),
                "privileged_count": int(np.sum(protected_mask == 0)),
            },
            "metrics": {
                "disparate_impact_ratio": dir_score,
                "equal_opportunity_difference": eod_score,
            },
            "four_fifths_rule_passed": is_dir_compliant,
            "equal_opportunity_passed": is_eod_compliant,
            "overall_fairness_approved": overall_compliant,
            "sr_11_7_status": "APPROVED" if overall_compliant else "REQUIRES_MITIGATION",
        }
