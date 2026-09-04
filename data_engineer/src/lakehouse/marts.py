"""
Gold Layer Analytical Marts.
Generates aggregated business intelligence views for telemarketing ROI and segment conversions.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from ..config import LakehouseConfig, load_lakehouse_config
from .db import DuckDBManager


class LakehouseMarts:
    def __init__(self, config: Optional[LakehouseConfig] = None):
        self.config = config or load_lakehouse_config()
        self.gold_dir = self.config.get_storage_path("gold")
        self.db = DuckDBManager(self.config.abs_db_path)

    def build_marts(self) -> Dict[str, int]:
        """
        Query Silver tables via DuckDB and generate Gold Data Marts:
          - mart_campaign_performance
          - mart_customer_segment_conversion
        """
        cust_file = self.config.get_storage_path("silver") / f"{self.config.tables.silver_customer}.parquet"
        fact_file = self.config.get_storage_path("silver") / f"{self.config.tables.silver_interaction}.parquet"

        if not cust_file.exists() or not fact_file.exists():
            from .transformation import LakehouseTransformation
            trans = LakehouseTransformation(self.config)
            trans.process_silver()

        cust_path = str(cust_file).replace("\\", "/")
        fact_path = str(fact_file).replace("\\", "/")

        # 1. Campaign Performance Mart SQL
        campaign_mart_sql = f"""
            SELECT
                f.month,
                f.contact,
                CASE 
                    WHEN f.campaign = 1 THEN '1 call'
                    WHEN f.campaign BETWEEN 2 AND 3 THEN '2-3 calls'
                    ELSE '4+ calls'
                END AS campaign_pressure_bucket,
                COUNT(*) AS total_calls,
                ROUND(AVG(f.duration), 1) AS avg_duration_seconds,
                SUM(CASE WHEN f.y = 'yes' THEN 1 ELSE 0 END) AS total_conversions,
                ROUND(SUM(CASE WHEN f.y = 'yes' THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) AS conversion_rate
            FROM read_parquet('{fact_path}') f
            GROUP BY 1, 2, 3
            ORDER BY total_conversions DESC
        """
        df_campaign_mart = self.db.query_df(campaign_mart_sql)

        # 2. Customer Segment Conversion Mart SQL
        segment_mart_sql = f"""
            SELECT
                c.job,
                c.education,
                c.housing,
                c.loan,
                COUNT(*) AS total_customers,
                ROUND(AVG(c.balance), 2) AS avg_balance_eur,
                SUM(CASE WHEN f.y = 'yes' THEN 1 ELSE 0 END) AS total_conversions,
                ROUND(SUM(CASE WHEN f.y = 'yes' THEN 1.0 ELSE 0.0 END) / COUNT(*), 4) AS conversion_rate
            FROM read_parquet('{cust_path}') c
            JOIN read_parquet('{fact_path}') f ON c.customer_id = f.customer_id
            GROUP BY 1, 2, 3, 4
            ORDER BY conversion_rate DESC
        """
        df_segment_mart = self.db.query_df(segment_mart_sql)

        # Save to Gold Parquet
        camp_file = self.gold_dir / f"{self.config.tables.gold_campaign_mart}.parquet"
        seg_file = self.gold_dir / f"{self.config.tables.gold_segment_mart}.parquet"

        df_campaign_mart.to_parquet(camp_file, index=False)
        df_segment_mart.to_parquet(seg_file, index=False)

        # Register Gold Views in DuckDB
        self.db.register_parquet(self.config.tables.gold_campaign_mart, camp_file)
        self.db.register_parquet(self.config.tables.gold_segment_mart, seg_file)

        return {
            "campaign_mart_rows": len(df_campaign_mart),
            "segment_mart_rows": len(df_segment_mart),
        }
