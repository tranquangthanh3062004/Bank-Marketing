"""
AI Feature Store module.
Provides Offline Feature Store (point-in-time training sets)
and Online Feature Store (fast sub-millisecond retrieval for inference).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from ..config import LakehouseConfig, load_lakehouse_config
from .db import DuckDBManager


class FeatureStore:
    def __init__(self, config: Optional[LakehouseConfig] = None):
        self.config = config or load_lakehouse_config()
        self.gold_dir = self.config.get_storage_path("gold")
        self.db = DuckDBManager(self.config.abs_db_path)
        self._online_cache: Dict[str, Dict[str, Any]] = {}

    def build_offline_feature_store(self) -> pd.DataFrame:
        """
        Build Offline Feature Store combining Silver tables with derived engineering logic.
        Generates clean, leakage-free feature vectors for AI training.
        """
        silver_cust = self.config.tables.silver_customer
        silver_fact = self.config.tables.silver_interaction

        feature_sql = f"""
            SELECT
                c.customer_id,
                -- Demographics
                c.age,
                c.job,
                c.marital,
                c.education,
                -- Financial Profile
                c.default,
                c.balance,
                c.housing,
                c.loan,
                -- Campaign Interaction Facts
                f.contact,
                f.day,
                f.month,
                f.duration,
                f.campaign,
                f.pdays,
                f.previous,
                f.poutcome,
                -- Derived Engineering Features:
                CASE 
                    WHEN c.balance < 0 THEN 'negative'
                    WHEN c.balance BETWEEN 0 AND 1000 THEN 'low_0_1k'
                    WHEN c.balance BETWEEN 1001 AND 5000 THEN 'mid_1k_5k'
                    ELSE 'high_5k_plus'
                END AS balance_tier,
                CASE 
                    WHEN f.pdays = -1 THEN 'never'
                    WHEN f.pdays <= 30 THEN 'recent_le_30d'
                    WHEN f.pdays <= 180 THEN 'mid_31_180d'
                    ELSE 'long_ago_gt_180d'
                END AS pdays_group,
                (CASE WHEN c.default = 'yes' THEN 2 ELSE 0 END +
                 CASE WHEN c.housing = 'yes' THEN 1 ELSE 0 END +
                 CASE WHEN c.loan = 'yes' THEN 1 ELSE 0 END) AS financial_pressure_index,
                ROUND(f.campaign * 1.0 / (f.previous + 1), 3) AS campaign_pressure_ratio,
                CASE WHEN f.pdays != -1 THEN 1 ELSE 0 END AS is_previously_contacted,
                CASE WHEN f.poutcome = 'success' THEN 1 ELSE 0 END AS past_success_flag,
                f.y
            FROM {silver_cust} c
            JOIN {silver_fact} f ON c.customer_id = f.customer_id
        """
        df_features = self.db.query_df(feature_sql)

        # Save to Gold Layer
        out_file = self.gold_dir / f"{self.config.tables.feature_store_offline}.parquet"
        df_features.to_parquet(out_file, index=False, compression="snappy")

        # Register Feature View
        self.db.register_parquet(self.config.tables.feature_store_offline, out_file)

        # Populate in-memory online cache for demo/fast lookup
        self._populate_online_cache(df_features)

        return df_features

    def get_offline_features(self) -> pd.DataFrame:
        """Retrieve historical training feature dataset."""
        out_file = self.gold_dir / f"{self.config.tables.feature_store_offline}.parquet"
        if not out_file.exists():
            return self.build_offline_feature_store()
        return pd.read_parquet(out_file)

    def _populate_online_cache(self, df: pd.DataFrame) -> None:
        """Populate key-value online store in memory."""
        records = df.to_dict(orient="records")
        for r in records:
            cid = str(r.get("customer_id"))
            self._online_cache[cid] = r

    def get_online_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve features for a customer by ID."""
        if customer_id in self._online_cache:
            return self._online_cache[customer_id]
        
        # Fallback to DuckDB lookup
        try:
            sql = f"SELECT * FROM {self.config.tables.feature_store_offline} WHERE customer_id = ?"
            df = self.db.query_df(sql, [customer_id])
            if not df.empty:
                feat = df.iloc[0].to_dict()
                self._online_cache[customer_id] = feat
                return feat
        except Exception:
            pass
        return None

    @staticmethod
    def derive_lead_features(lead_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute derived Lakehouse features on-the-fly for real-time lead inference.
        """
        d = dict(lead_dict)

        # Standardize strings
        for k in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:
            if k in d and d[k] is not None:
                d[k] = str(d[k]).strip().lower()

        balance = float(d.get("balance", 0.0))
        if balance < 0:
            d["balance_tier"] = "negative"
        elif balance <= 1000:
            d["balance_tier"] = "low_0_1k"
        elif balance <= 5000:
            d["balance_tier"] = "mid_1k_5k"
        else:
            d["balance_tier"] = "high_5k_plus"

        pdays = int(d.get("pdays", -1))
        if pdays == -1:
            d["pdays_group"] = "never"
        elif pdays <= 30:
            d["pdays_group"] = "recent_le_30d"
        elif pdays <= 180:
            d["pdays_group"] = "mid_31_180d"
        else:
            d["pdays_group"] = "long_ago_gt_180d"

        # Financial pressure
        fin_idx = 0
        if str(d.get("default", "no")).lower() == "yes":
            fin_idx += 2
        if str(d.get("housing", "no")).lower() == "yes":
            fin_idx += 1
        if str(d.get("loan", "no")).lower() == "yes":
            fin_idx += 1
        d["financial_pressure_index"] = fin_idx

        # Campaign pressure
        campaign = int(d.get("campaign", 1))
        previous = int(d.get("previous", 0))
        d["campaign_pressure_ratio"] = round(campaign / (previous + 1), 3)

        d["is_previously_contacted"] = 1 if pdays != -1 else 0
        d["past_success_flag"] = 1 if str(d.get("poutcome", "")).lower() == "success" else 0

        return d
