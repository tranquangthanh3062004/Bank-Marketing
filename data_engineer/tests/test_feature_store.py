"""
Unit tests for AI Feature Store (Offline feature engineering & Online lookup).
"""

from data_engineer.src.config import load_lakehouse_config
from data_engineer.src.lakehouse.feature_store import FeatureStore


def test_build_and_get_offline_features():
    cfg = load_lakehouse_config()
    fs = FeatureStore(cfg)
    df_feat = fs.build_offline_feature_store()

    assert not df_feat.empty
    assert "balance_tier" in df_feat.columns
    assert "financial_pressure_index" in df_feat.columns
    assert "campaign_pressure_ratio" in df_feat.columns
    assert "past_success_flag" in df_feat.columns
    assert "y" in df_feat.columns

    # Test reading back
    df_read = fs.get_offline_features()
    assert len(df_read) == len(df_feat)


def test_derive_lead_features():
    lead = {
        "customer_id": "LEAD_TEST",
        "age": 52,
        "job": "management",
        "balance": 6500,
        "default": "no",
        "housing": "yes",
        "loan": "yes",
        "campaign": 2,
        "previous": 1,
        "pdays": 15,
        "poutcome": "success",
    }
    derived = FeatureStore.derive_lead_features(lead)

    assert derived["balance_tier"] == "high_5k_plus"
    assert derived["financial_pressure_index"] == 2  # housing(1) + loan(1)
    assert derived["campaign_pressure_ratio"] == 1.0  # 2 / (1 + 1)
    assert derived["is_previously_contacted"] == 1
    assert derived["past_success_flag"] == 1
    assert derived["pdays_group"] == "recent_le_30d"


def test_online_feature_lookup():
    cfg = load_lakehouse_config()
    fs = FeatureStore(cfg)
    df_feat = fs.get_offline_features()

    if not df_feat.empty:
        first_cid = df_feat.iloc[0]["customer_id"]
        feat = fs.get_online_features(first_cid)
        assert feat is not None
        assert feat["customer_id"] == first_cid
        assert "balance_tier" in feat
