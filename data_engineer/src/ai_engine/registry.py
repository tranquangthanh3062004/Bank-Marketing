"""
Model Registry and Artifact Store.
Manages versioning, metadata logging, and persistence of model artifacts.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
from ..config import ModelConfig, load_model_config
from .explainability import ModelExplainer
from .features import FeaturePipeline


class ModelRegistry:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.registry_dir = self.config.abs_registry_dir

    def save_model(
        self,
        model: Any,
        pipeline: FeaturePipeline,
        metrics: Dict[str, Any],
        threshold_info: Dict[str, Any],
        global_importance: Optional[list] = None,
        model_type: str = "lightgbm",
        mode: str = "post_call",
    ) -> Dict[str, str]:
        """Save model artifacts and metadata to registry with mode support."""
        prefix = f"{mode}_{model_type}" if mode == "pre_call" else model_type
        model_file = self.registry_dir / f"{prefix}_model.joblib"
        preprocessor_file = self.registry_dir / f"{prefix}_preprocessor.joblib"
        metadata_file = (
            self.registry_dir / "pre_call_metadata.json"
            if mode == "pre_call"
            else self.registry_dir / "model_metadata.json"
        )

        # Save artifacts
        joblib.dump(model, model_file)
        joblib.dump(pipeline, preprocessor_file)

        # Save active model pointers
        if mode == "pre_call":
            joblib.dump(model, self.registry_dir / "pre_call_best_model.joblib")
            joblib.dump(pipeline, self.registry_dir / "pre_call_best_preprocessor.joblib")
        else:
            joblib.dump(model, self.registry_dir / "best_model.joblib")
            joblib.dump(pipeline, self.registry_dir / "best_preprocessor.joblib")

        metadata = {
            "name": self.config.name,
            "version": self.config.version,
            "model_type": model_type,
            "mode": mode,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "optimal_threshold": threshold_info.get("optimal_threshold", 0.5),
            "threshold_metrics": threshold_info,
            "test_metrics": metrics,
            "global_importance": global_importance or [],
            "feature_names": pipeline.feature_names,
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "model_path": str(model_file),
            "preprocessor_path": str(preprocessor_file),
            "metadata_path": str(metadata_file),
        }

    def load_active_model(self, mode: str = "post_call") -> Tuple[Any, FeaturePipeline, Dict[str, Any]]:
        """Load the active model, preprocessor, and metadata for specified mode."""
        if mode == "pre_call" and (self.registry_dir / "pre_call_best_model.joblib").exists():
            model_file = self.registry_dir / "pre_call_best_model.joblib"
            prep_file = self.registry_dir / "pre_call_best_preprocessor.joblib"
            meta_file = self.registry_dir / "pre_call_metadata.json"
        else:
            model_file = self.registry_dir / "best_model.joblib"
            prep_file = self.registry_dir / "best_preprocessor.joblib"
            meta_file = self.registry_dir / "model_metadata.json"

        if not model_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"No registered model found in {self.registry_dir} for mode '{mode}'. Please train a model first!"
            )

        model = joblib.load(model_file)
        pipeline = joblib.load(prep_file)

        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return model, pipeline, metadata

    def is_model_available(self, mode: str = "post_call") -> bool:
        """Check if a trained model is ready in registry for specified mode."""
        if mode == "pre_call":
            return (
                (self.registry_dir / "pre_call_best_model.joblib").exists()
                and (self.registry_dir / "pre_call_metadata.json").exists()
            )
        return (
            (self.registry_dir / "best_model.joblib").exists()
            and (self.registry_dir / "model_metadata.json").exists()
        )
