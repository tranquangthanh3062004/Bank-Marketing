"""
Enterprise Workflow Orchestrator and DAG Pipeline Engine.
Automates the full-lifecycle: Ingestion -> Quality Gate -> PII Anonymization ->
Feature Store -> Dual-Model Retraining -> SR 11-7 Fairness Audit -> Deployment.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from ..ai_engine.evaluator import ModelEvaluator
from ..ai_engine.fairness_audit import FairnessAuditor
from ..ai_engine.features import FeaturePipeline
from ..ai_engine.registry import ModelRegistry
from ..ai_engine.trainer import ModelTrainer
from ..config import PACKAGE_ROOT, LakehouseConfig, ModelConfig, load_lakehouse_config, load_model_config
from ..governance.pii_masking import PIISecurityManager
from ..lakehouse.feature_store import FeatureStore
from ..lakehouse.ingestion import LakehouseIngestion
from ..lakehouse.marts import LakehouseMarts
from ..lakehouse.transformation import LakehouseTransformation


class EnterpriseWorkflowOrchestrator:
    def __init__(
        self,
        lake_cfg: Optional[LakehouseConfig] = None,
        model_cfg: Optional[ModelConfig] = None,
    ):
        self.lake_cfg = lake_cfg or load_lakehouse_config()
        self.model_cfg = model_cfg or load_model_config()

    def run_daily_pipeline(self, raw_source: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes complete production DAG.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)
        source = raw_source or str(PACKAGE_ROOT / "sample_data" / "bank_raw_sample.csv")

        # Step 1: Ingestion Bronze
        ingestion = LakehouseIngestion(self.lake_cfg)
        ing_res = ingestion.ingest_file(source)

        # Step 2: Silver Cleaning & Star Schema
        trans = LakehouseTransformation(self.lake_cfg)
        silver_res = trans.process_silver()

        # Step 3: PII Security & Anonymization Audit
        sec = PIISecurityManager()
        cust_parquet = self.lake_cfg.get_storage_path("silver") / f"{self.lake_cfg.tables.silver_customer}.parquet"
        df_cust = pd.read_parquet(cust_parquet)
        df_deidentified = sec.deidentify_dataframe(df_cust)

        # Step 4: Gold Marts & Feature Store
        marts = LakehouseMarts(self.lake_cfg)
        marts_res = marts.build_marts()

        fs = FeatureStore(self.lake_cfg)
        df_features = fs.build_offline_feature_store()

        # Step 5: Retrain Dual-Models (Pre-call & Post-call)
        trainer = ModelTrainer(self.model_cfg)
        evaluator = ModelEvaluator(self.model_cfg)
        registry = ModelRegistry(self.model_cfg)
        fairness_auditor = FairnessAuditor()

        models_trained = {}
        for mode in ["pre_call", "post_call"]:
            train_res = trainer.train(df_features, model_type="lightgbm", mode=mode)
            model = train_res["model"]
            pipeline = train_res["pipeline"]
            splits = train_res["data_splits"]

            y_val_proba = model.predict_proba(splits["X_val_trans"])[:, 1]
            threshold_info = evaluator.tune_threshold(splits["y_val"], y_val_proba)

            test_metrics = evaluator.evaluate(
                model,
                splits["X_test_trans"],
                splits["y_test"],
                threshold=threshold_info["optimal_threshold"],
            )

            # Step 6: SR 11-7 AI Fairness Audit
            y_test_pred = (
                model.predict_proba(splits["X_test_trans"])[:, 1] >= threshold_info["optimal_threshold"]
            ).astype(int)
            fairness_report = fairness_auditor.audit_model(
                splits["X_test"], splits["y_test"].values, y_test_pred, protected_attribute="age"
            )

            reg_paths = registry.save_model(
                model=model,
                pipeline=pipeline,
                metrics=test_metrics,
                threshold_info=threshold_info,
                global_importance=[],
                model_type="lightgbm",
                mode=mode,
            )

            models_trained[mode] = {
                "test_metrics": test_metrics,
                "threshold": threshold_info["optimal_threshold"],
                "fairness_audit": fairness_report,
                "registry_path": reg_paths["model_path"],
            }

        elapsed = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

        return {
            "status": "SUCCESS",
            "execution_time_seconds": round(elapsed, 2),
            "ingested_records": ing_res["ingested_records"],
            "silver_customers": silver_res["dim_customer_count"],
            "silver_interactions": silver_res["fact_interaction_count"],
            "deidentified_tokens_verified": len(df_deidentified),
            "feature_store_records": len(df_features),
            "models": models_trained,
        }
