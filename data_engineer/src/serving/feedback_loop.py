"""
Closed-Loop CRM & CTI Feedback Loop Module.
Receives post-call outcomes from Telesales agents (agreed/declined/callback/unreachable)
and writes them back into the Lakehouse Silver fact table for continuous learning.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from ..config import LakehouseConfig, load_lakehouse_config
from ..lakehouse.db import DuckDBManager


class CallFeedbackPayload(BaseModel):
    customer_id: str = Field(description="Mã định danh khách hàng")
    agent_id: str = Field(default="AGENT_001", description="Mã tổng đài viên")
    call_duration: int = Field(ge=0, description="Thời lượng cuộc gọi thực tế (giây)")
    outcome: Literal["agreed", "declined", "callback", "unreachable"] = Field(
        description="Kết quả cuộc gọi thực tế"
    )
    product_type: str = Field(default="term_deposit", description="Sản phẩm tư vấn")
    notes: Optional[str] = Field(default=None, description="Ghi chú của tổng đài viên")


class FeedbackLoopManager:
    def __init__(self, config: Optional[LakehouseConfig] = None):
        self.config = config or load_lakehouse_config()
        self.silver_dir = self.config.get_storage_path("silver")
        self.db = DuckDBManager(self.config.abs_db_path)

    def record_feedback(self, payload: CallFeedbackPayload) -> Dict[str, Any]:
        """
        Record post-call feedback and append to feedback log parquet.
        """
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        mapped_y = "yes" if payload.outcome == "agreed" else "no"

        feedback_record = {
            "feedback_id": f"FB_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
            "customer_id": payload.customer_id,
            "agent_id": payload.agent_id,
            "call_duration": payload.call_duration,
            "outcome": payload.outcome,
            "y_label": mapped_y,
            "notes": payload.notes or "",
            "recorded_at": now_ts,
        }

        # Save to silver feedback partition
        feedback_file = self.silver_dir / "crm_call_feedback.parquet"
        df_new = pd.DataFrame([feedback_record])

        if feedback_file.exists():
            df_existing = pd.read_parquet(feedback_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_parquet(feedback_file, index=False, compression="snappy")

        return {
            "status": "RECORDED",
            "feedback_id": feedback_record["feedback_id"],
            "customer_id": payload.customer_id,
            "mapped_y": mapped_y,
            "total_feedbacks_recorded": len(df_combined),
        }
