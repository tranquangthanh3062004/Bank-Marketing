"""
Batch Lead Scoring Engine for Telemarketing Campaigns.
Scores prospect lists, maps calibrated probabilities to Priority Tiers, and outputs dialer queues.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
from ..ai_engine.registry import ModelRegistry
from ..config import ModelConfig, load_model_config
from ..lakehouse.feature_store import FeatureStore


class BatchLeadScorer:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.registry = ModelRegistry(self.config)
        self.model, self.pipeline, self.metadata = self.registry.load_active_model()
        self.optimal_threshold = self.metadata.get("optimal_threshold", 0.5)

    def _assign_tier(self, prob: float) -> str:
        """Assign telemarketing priority tier based on conversion probability."""
        tiers = self.config.serving.get("priority_tiers", {})
        t1 = tiers.get("tier_1_threshold", 0.70)
        t2 = tiers.get("tier_2_threshold", 0.45)
        t3 = tiers.get("tier_3_threshold", 0.25)

        if prob >= t1:
            return "Tier 1 (Hot - Priority Dispatch)"
        elif prob >= t2:
            return "Tier 2 (Warm - Follow Up)"
        elif prob >= t3:
            return "Tier 3 (Neutral - Low Priority)"
        else:
            return "Tier 4 (Cold - Do Not Call)"

    def _assign_recommendation(self, tier: str) -> str:
        if "Tier 1" in tier:
            return "Chuyển ngay cho Senior Telesales; gọi trong 30 phút; giới thiệu gói lãi suất ưu đãi"
        elif "Tier 2" in tier:
            return "Gán cho đội Telesales tiêu chuẩn; gọi vào khung giờ nghỉ trưa hoặc sau 17h"
        elif "Tier 3" in tier:
            return "Gửi tin nhắn SMS / Email giới thiệu trước; chỉ gọi lại nếu khách hàng mở link"
        else:
            return "Loại khỏi danh sách gọi chiến dịch này để tiết kiệm chi phí vận hành"

    def score_dataframe(self, df_leads: pd.DataFrame) -> pd.DataFrame:
        """Score dataframe of leads and append probability, tier, and recommendations."""
        df_out = df_leads.copy()

        # Derive required feature store columns
        records = df_out.to_dict(orient="records")
        derived_records = [FeatureStore.derive_lead_features(r) for r in records]
        df_features = pd.DataFrame(derived_records)

        # Transform using fitted pipeline
        X_trans = self.pipeline.transform(df_features)

        # Predict probability
        probabilities = self.model.predict_proba(X_trans)[:, 1]

        df_out["predicted_conversion_prob"] = np.round(probabilities, 4)
        df_out["is_recommended_call"] = (probabilities >= self.optimal_threshold).astype(int)
        df_out["priority_tier"] = [self._assign_tier(p) for p in probabilities]
        df_out["telesales_action"] = [self._assign_recommendation(t) for t in df_out["priority_tier"]]

        # Sort by highest potential first
        df_out = df_out.sort_values(by="predicted_conversion_prob", ascending=False).reset_index(drop=True)
        return df_out

    def score_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> pd.DataFrame:
        """Read input leads file, score, and write back."""
        inp_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if inp_p.suffix.lower() == ".parquet":
            df = pd.read_parquet(inp_p)
        else:
            df = pd.read_csv(inp_p)

        scored_df = self.score_dataframe(df)

        if out_p.suffix.lower() == ".parquet":
            scored_df.to_parquet(out_p, index=False)
        else:
            scored_df.to_csv(out_p, index=False)

        return scored_df
