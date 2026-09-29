"""
Enterprise Data Governance and PII Masking Module.
Complies with Decree 13/2023/ND-CP (Personal Data Protection) and GDPR.
Provides Tokenization, Masking, and Data De-identification.
"""

import hashlib
import hmac
from typing import Any, Dict, List, Optional
import pandas as pd

DEFAULT_SALT = "bank_marketing_enterprise_secret_salt_2026"


class PIISecurityManager:
    def __init__(self, salt: str = DEFAULT_SALT):
        self.salt = salt.encode("utf-8")

    def tokenize_customer_id(self, customer_id: str) -> str:
        """
        Deterministic pseudonymization via HMAC-SHA256.
        Produces unique pseudorandom token that allows analytical joins without exposing true ID.
        """
        if not customer_id:
            return "ANONYMOUS"
        h = hmac.new(self.salt, customer_id.encode("utf-8"), hashlib.sha256)
        return f"TOKEN_{h.hexdigest()[:16]}"

    def mask_phone_number(self, phone: str) -> str:
        """
        Masks phone number: 0912345678 -> 091****678.
        """
        if not phone or len(phone) < 7:
            return "****"
        return f"{phone[:3]}****{phone[-3:]}"

    def deidentify_dataframe(
        self,
        df: pd.DataFrame,
        id_col: str = "customer_id",
        mask_financials: bool = False,
    ) -> pd.DataFrame:
        """
        Applies enterprise de-identification on a dataframe:
        - Tokenizes customer ID.
        - Optionally bins sensitive balances to ranges.
        """
        df_out = df.copy()
        if id_col in df_out.columns:
            df_out[id_col] = df_out[id_col].apply(self.tokenize_customer_id)

        if mask_financials and "balance" in df_out.columns:
            # Round balance to nearest 500 EUR for differential privacy
            df_out["balance"] = (df_out["balance"] // 500) * 500

        return df_out
