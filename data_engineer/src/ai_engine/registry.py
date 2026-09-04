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
    ) -> Dict[str, str]:
        """Save model artifacts and metadata to registry."""
        model_file = self.registry_dir / f"{model_type}_model.joblib"
        preprocessor_file = self.registry_dir / f"{model_type}_preprocessor.joblib"
        metadata_file = self.registry_dir / "model_metadata.json"

        # Save artifacts
        joblib.dump(model, model_file)
        joblib.dump(pipeline, preprocessor_file)

        # Save active best model pointer
        best_model_file = self.registry_dir / "best_model.joblib"
        best_prep_file = self.registry_dir / "best_preprocessor.joblib"
        joblib.dump(model, best_model_file)
        joblib.dump(pipeline, best_prep_file)

        metadata = {
            "name": self.config.name,
            "version": self.config.version,
            "model_type": model_type,
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

    def load_active_model(self) -> Tuple[Any, FeaturePipeline, Dict[str, Any]]:
        """Load the active best model, its preprocessor, and metadata."""
        best_model_file = self.registry_dir / "best_model.joblib"
        best_prep_file = self.registry_dir / "best_preprocessor.joblib"
        metadata_file = self.registry_dir / "model_metadata.json"

        if not best_model_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"No registered model found in {self.registry_dir}. Please train a model first!"
            )

        model = joblib.load(best_model_file)
        pipeline = joblib.load(best_prep_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return model, pipeline, metadata

    def is_model_available(self) -> bool:
        """Check if a trained model is ready in registry."""
        best_model_file = self.registry_dir / "best_model.joblib"
        metadata_file = self.registry_dir / "model_metadata.json"
        return best_model_file.exists() and metadata_file.exists()
