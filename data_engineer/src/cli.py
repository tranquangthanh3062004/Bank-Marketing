"""
Unified Command Line Interface (CLI) for Data Lakehouse & AI Engine.
Provides commands for Lakehouse ELT, Feature Store, Model Training, Scoring, and API Serving.
"""

import argparse
import sys
from pathlib import Path
from .ai_engine.evaluator import ModelEvaluator
from .ai_engine.explainability import ModelExplainer
from .ai_engine.registry import ModelRegistry
from .ai_engine.trainer import ModelTrainer
from .config import PACKAGE_ROOT, load_lakehouse_config, load_model_config
from .lakehouse.feature_store import FeatureStore
from .lakehouse.ingestion import LakehouseIngestion
from .lakehouse.marts import LakehouseMarts
from .lakehouse.transformation import LakehouseTransformation
from .serving.batch_scorer import BatchLeadScorer


def cmd_ingest(args):
    print(f"🚀 [Lakehouse] Ingesting raw data from: {args.source}")
    ingestion = LakehouseIngestion()
    res = ingestion.ingest_file(args.source)
    print("✅ Ingestion Completed!")
    print(f"   Batch ID: {res['batch_id']}")
    print(f"   Ingested: {res['ingested_records']} records")
    print(f"   Quarantined: {res['quarantined_records']} records")
    print(f"   Bronze Parquet: {res['bronze_file']}")


def cmd_transform_silver(args):
    print("🚀 [Lakehouse] Transforming Bronze -> Silver (Cleaning & Conforming Star Schema)...")
    trans = LakehouseTransformation()
    res = trans.process_silver()
    print("✅ Silver Transformation Completed!")
    print(f"   Conformed dim_customer: {res['dim_customer_count']} rows")
    print(f"   Conformed fact_campaign_interaction: {res['fact_interaction_count']} rows")


def cmd_transform_gold(args):
    print("🚀 [Lakehouse] Generating Gold Analytical Marts...")
    marts = LakehouseMarts()
    res = marts.build_marts()
    print("✅ Gold Marts Generation Completed!")
    print(f"   Campaign Performance Mart: {res['campaign_mart_rows']} rows")
    print(f"   Customer Segment Mart: {res['segment_mart_rows']} rows")


def cmd_build_features(args):
    print("🚀 [Lakehouse] Building Offline & Online Feature Store...")
    fs = FeatureStore()
    df_feat = fs.build_offline_feature_store()
    print(f"✅ Feature Store Built! Generated {len(df_feat)} rows with {len(df_feat.columns)} feature columns.")


def cmd_run_lakehouse(args):
    print("=======================================================")
    print("      DATA LAKEHOUSE MEDALLION PIPELINE EXECUTION      ")
    print("=======================================================")
    source = args.source or str(PACKAGE_ROOT / "sample_data" / "bank_raw_sample.csv")
    args.source = source
    cmd_ingest(args)
    cmd_transform_silver(args)
    cmd_transform_gold(args)
    cmd_build_features(args)
    print("🎉 All Lakehouse Layers (Bronze -> Silver -> Gold -> Features) successfully processed!")


def cmd_train(args):
    print(f"🤖 [AI Engine] Training model type: {args.model}")
    fs = FeatureStore()
    df_features = fs.get_offline_features()

    trainer = ModelTrainer()
    train_res = trainer.train(df_features, model_type=args.model)

    model = train_res["model"]
    pipeline = train_res["pipeline"]
    splits = train_res["data_splits"]

    # Evaluate & tune threshold
    evaluator = ModelEvaluator()
    y_val_proba = model.predict_proba(splits["X_val_trans"])[:, 1]
    threshold_info = evaluator.tune_threshold(splits["y_val"], y_val_proba)

    # Test evaluation
    test_metrics = evaluator.evaluate(
        model,
        splits["X_test_trans"],
        splits["y_test"],
        threshold=threshold_info["optimal_threshold"],
    )

    # Global SHAP importance
    explainer = ModelExplainer(model, pipeline, model_type=args.model)
    global_importance = explainer.get_global_importance(splits["X_train_trans"], top_n=8)

    # Register model
    registry = ModelRegistry()
    reg_paths = registry.save_model(
        model=model,
        pipeline=pipeline,
        metrics=test_metrics,
        threshold_info=threshold_info,
        global_importance=global_importance,
        model_type=args.model,
    )

    print("✅ Model Training & Registration Completed!")
    print(f"   ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"   F1-Score: {test_metrics['f1_score']:.4f} (at Optimal Threshold {test_metrics['threshold_used']})")
    print(f"   Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f}")
    print("   Top Influential Features:")
    for feat in global_importance[:5]:
        print(f"     - {feat['feature']}: {feat['importance']}")
    print(f"   Artifacts saved to: {registry.registry_dir}")


def cmd_batch_score(args):
    print(f"🎯 [Serving] Batch Lead Scoring on: {args.input}")
    output = args.output or str(PACKAGE_ROOT / "sample_data" / "scored_leads_output.csv")
    scorer = BatchLeadScorer()
    scored_df = scorer.score_file(args.input, output)

    print(f"✅ Scored {len(scored_df)} leads successfully! Saved to: {output}")
    print("   Priority Breakdown:")
    tier_counts = scored_df["priority_tier"].value_counts()
    for tier, count in tier_counts.items():
        print(f"     - {tier}: {count} leads")


def cmd_serve(args):
    import uvicorn
    print(f"🌐 [Serving] Launching FastAPI Lead Scoring Service on http://{args.host}:{args.port} ...")
    uvicorn.run("data_engineer.src.serving.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_run_all(args):
    print("==================================================================")
    print("   FULL END-TO-END DATA LAKEHOUSE + AI ENGINE PIPELINE RUNNER   ")
    print("==================================================================")
    cmd_run_lakehouse(args)
    print("\n------------------------------------------------------------------")
    cmd_train(args)
    print("\n------------------------------------------------------------------")
    args.input = str(PACKAGE_ROOT / "sample_data" / "leads_to_score.csv")
    args.output = str(PACKAGE_ROOT / "sample_data" / "scored_leads_output.csv")
    cmd_batch_score(args)
    print("\n🎉 Complete Lakehouse & AI Engine Pipeline successfully executed from end-to-end!")


def main():
    parser = argparse.ArgumentParser(
        prog="data-engineer",
        description="Bank Marketing Data Lakehouse & AI Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest raw file to Bronze layer")
    p_ingest.add_argument("--source", required=True, help="Path to raw source file (CSV or Parquet)")

    # Silver
    subparsers.add_parser("transform-silver", help="Process Bronze to Silver layer")

    # Gold
    subparsers.add_parser("transform-gold", help="Build Gold analytical marts")

    # Feature store
    subparsers.add_parser("build-features", help="Build Offline & Online Feature Store")

    # Run Lakehouse
    p_lake = subparsers.add_parser("run-lakehouse", help="Execute complete Lakehouse Bronze->Silver->Gold->Features")
    p_lake.add_argument("--source", default=None, help="Optional raw source path")

    # Train
    p_train = subparsers.add_parser("train", help="Train model and register artifacts")
    p_train.add_argument("--model", default="lightgbm", choices=["lightgbm", "logistic_regression"], help="Model type")

    # Batch score
    p_score = subparsers.add_parser("batch-score", help="Batch score leads for telemarketing")
    p_score.add_argument("--input", required=True, help="Input CSV/Parquet leads file")
    p_score.add_argument("--output", default=None, help="Output CSV/Parquet path")

    # Serve API
    p_serve = subparsers.add_parser("serve", help="Run FastAPI Lead Scoring server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host address")
    p_serve.add_argument("--port", default=8000, type=int, help="Port number")
    p_serve.add_argument("--reload", action="store_true", help="Enable reload")

    # Run All
    p_all = subparsers.add_parser("run-all", help="Run full pipeline: Lakehouse + AI Engine + Batch Score")
    p_all.add_argument("--source", default=None, help="Optional raw source path")
    p_all.add_argument("--model", default="lightgbm", choices=["lightgbm", "logistic_regression"])

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "ingest": cmd_ingest,
        "transform-silver": cmd_transform_silver,
        "transform-gold": cmd_transform_gold,
        "build-features": cmd_build_features,
        "run-lakehouse": cmd_run_lakehouse,
        "train": cmd_train,
        "batch-score": cmd_batch_score,
        "serve": cmd_serve,
        "run-all": cmd_run_all,
    }

    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
