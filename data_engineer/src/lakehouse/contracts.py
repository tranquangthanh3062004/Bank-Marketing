"""
Data Quality Contracts using Pandera and Pydantic.
Enforces schema validation and guards against data corruption or drift.
"""

from typing import List, Literal, Optional, Tuple
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from pydantic import BaseModel, Field, field_validator


# -------------------------------------------------------------------------
# Pydantic Model for Individual Real-Time API Payloads
# -------------------------------------------------------------------------
class CustomerLeadPayload(BaseModel):
    customer_id: Optional[str] = Field(default="ANONYMOUS", description="Mã định danh khách hàng")
    age: int = Field(ge=18, le=105, description="Độ tuổi khách hàng (18-105)")
    job: str = Field(description="Nghề nghiệp")
    marital: str = Field(description="Tình trạng hôn nhân (married, single, divorced)")
    education: str = Field(description="Trình độ học vấn (primary, secondary, tertiary, unknown)")
    default: str = Field(description="Có nợ xấu không (yes, no)")
    balance: float = Field(description="Số dư tài khoản bình quân (Euro)")
    housing: str = Field(description="Có khoản vay mua nhà (yes, no)")
    loan: str = Field(description="Có khoản vay cá nhân (yes, no)")
    contact: Optional[str] = Field(default="unknown", description="Kênh liên hệ (cellular, telephone, unknown)")
    day: Optional[int] = Field(default=15, ge=1, le=31, description="Ngày liên hệ trong tháng")
    month: Optional[str] = Field(default="may", description="Tháng liên hệ (jan, feb,..., dec)")
    duration: Optional[int] = Field(default=0, ge=0, description="Thời lượng cuộc gọi gần nhất (giây)")
    campaign: Optional[int] = Field(default=1, ge=1, description="Số lần liên hệ trong chiến dịch hiện tại")
    pdays: Optional[int] = Field(default=-1, ge=-1, description="Số ngày từ lần liên hệ trước (-1: chưa từng)")
    previous: Optional[int] = Field(default=0, ge=0, description="Số lần liên hệ trước chiến dịch này")
    poutcome: Optional[str] = Field(default="unknown", description="Kết quả chiến dịch trước")

    @field_validator("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
    @classmethod
    def clean_strings(cls, v: str) -> str:
        return str(v).strip().lower()


# -------------------------------------------------------------------------
# Pandera Schemas for Lakehouse Layers
# -------------------------------------------------------------------------
valid_jobs = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
]
valid_maritals = ["married", "single", "divorced"]
valid_educations = ["primary", "secondary", "tertiary", "unknown"]
valid_binaries = ["yes", "no"]
valid_contacts = ["cellular", "telephone", "unknown"]
valid_months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
valid_poutcomes = ["unknown", "failure", "other", "success"]

# Bronze Schema (Schema-on-readiness check)
bronze_schema = DataFrameSchema(
    columns={
        "age": Column(int, Check.in_range(18, 110), nullable=False),
        "balance": Column(float, nullable=False, coerce=True),
        "duration": Column(float, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
        "campaign": Column(int, Check.greater_than_or_equal_to(1), nullable=False, coerce=True),
        "pdays": Column(int, Check.greater_than_or_equal_to(-1), nullable=False, coerce=True),
        "previous": Column(int, Check.greater_than_or_equal_to(0), nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
)

# Silver Customer Dimension Schema
silver_customer_schema = DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False, unique=True),
        "age": Column(int, Check.in_range(18, 105), nullable=False),
        "job": Column(str, Check.isin(valid_jobs), nullable=False),
        "marital": Column(str, Check.isin(valid_maritals), nullable=False),
        "education": Column(str, Check.isin(valid_educations), nullable=False),
        "default": Column(str, Check.isin(valid_binaries), nullable=False),
        "balance": Column(float, nullable=False),
        "housing": Column(str, Check.isin(valid_binaries), nullable=False),
        "loan": Column(str, Check.isin(valid_binaries), nullable=False),
    },
    strict=False,
    coerce=True,
)

# Silver Campaign Interaction Fact Schema
silver_interaction_schema = DataFrameSchema(
    columns={
        "interaction_id": Column(str, nullable=False, unique=True),
        "customer_id": Column(str, nullable=False),
        "contact": Column(str, Check.isin(valid_contacts), nullable=False),
        "day": Column(int, Check.in_range(1, 31), nullable=False),
        "month": Column(str, Check.isin(valid_months), nullable=False),
        "duration": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "campaign": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
        "pdays": Column(int, Check.greater_than_or_equal_to(-1), nullable=False),
        "previous": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "poutcome": Column(str, Check.isin(valid_poutcomes), nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_bronze_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate raw data for Bronze.
    Returns:
        (valid_df, quarantine_df)
    """
    df_copy = df.copy()
    # Normalize column names
    df_copy.columns = [c.strip().lower() for c in df_copy.columns]

    # Required core columns
    required_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    missing = [c for c in required_cols if c not in df_copy.columns]
    if missing:
        raise ValueError(f"Missing mandatory columns for Bronze Lakehouse: {missing}")

    # Check basic integrity conditions
    is_valid_age = df_copy["age"].between(18, 105)
    is_valid_duration = df_copy["duration"] >= 0
    is_valid_campaign = df_copy["campaign"] >= 1

    valid_mask = is_valid_age & is_valid_duration & is_valid_campaign

    valid_df = df_copy[valid_mask].copy()
    quarantine_df = df_copy[~valid_mask].copy()

    if not quarantine_df.empty:
        quarantine_df["_quarantine_reason"] = "Failed range check (age 18-105, duration>=0, campaign>=1)"

    return valid_df, quarantine_df


def validate_silver_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Validate conformed Silver customer dimension."""
    return silver_customer_schema.validate(df)


def validate_silver_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Validate conformed Silver campaign interaction fact."""
    return silver_interaction_schema.validate(df)
