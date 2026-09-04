"""
Unit tests for data contracts (Pandera schemas and Pydantic models).
"""

import pandas as pd
import pytest
from pydantic import ValidationError
from data_engineer.src.lakehouse.contracts import (
    CustomerLeadPayload,
    validate_bronze_data,
    validate_silver_customer,
    validate_silver_interaction,
)


def test_customer_lead_payload_valid():
    payload = CustomerLeadPayload(
        customer_id="CUST_TEST",
        age=35,
        job="technician",
        marital="single",
        education="tertiary",
        default="no",
        balance=1500.0,
        housing="yes",
        loan="no",
        contact="cellular",
        duration=240,
        campaign=2,
        pdays=90,
        previous=1,
        poutcome="success",
    )
    assert payload.age == 35
    assert payload.job == "technician"
    assert payload.balance == 1500.0


def test_customer_lead_payload_invalid_age():
    with pytest.raises(ValidationError):
        CustomerLeadPayload(
            age=15,  # Must be >= 18
            job="student",
            marital="single",
            education="secondary",
            default="no",
            balance=100.0,
            housing="no",
            loan="no",
        )


def test_validate_bronze_data_quarantine():
    # Construct test dataframe with 1 valid row and 1 invalid row (negative duration)
    df = pd.DataFrame([
        {
            "age": 45,
            "job": "admin.",
            "balance": 2000.0,
            "duration": 180.0,
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
        },
        {
            "age": 45,
            "job": "admin.",
            "balance": 2000.0,
            "duration": -50.0,  # INVALID
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
        },
    ])
    valid_df, quarantine_df = validate_bronze_data(df)
    assert len(valid_df) == 1
    assert len(quarantine_df) == 1
    assert "_quarantine_reason" in quarantine_df.columns


def test_validate_silver_schemas():
    df_cust = pd.DataFrame([{
        "customer_id": "C_01",
        "age": 40,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 5000.0,
        "housing": "no",
        "loan": "no",
    }])
    validated_cust = validate_silver_customer(df_cust)
    assert len(validated_cust) == 1

    df_fact = pd.DataFrame([{
        "interaction_id": "I_01",
        "customer_id": "C_01",
        "contact": "cellular",
        "day": 12,
        "month": "may",
        "duration": 300.0,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
        "y": "yes",
    }])
    validated_fact = validate_silver_interaction(df_fact)
    assert len(validated_fact) == 1
