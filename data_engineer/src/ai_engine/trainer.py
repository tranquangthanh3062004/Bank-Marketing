"""
Model Trainer for Bank Marketing Term Deposit Prediction.
Supports Logistic Regression baseline and LightGBM with Imbalance Handling.
"""

from typing import Any, Dict, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ..config import ModelConfig, load_model_config
from .features import FeaturePipeline


class ModelTrainer:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()

    def prepare_data_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split feature dataframe into Train, Validation, and Test partitions.
        Target is converted to binary 0/1.
        """
        target_col = self.config.dataset["target_column"]
        pos_label = self.config.dataset["positive_label"]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset!")

        y_binary = (df[target_col].astype(str).str.lower() == pos_label.lower()).astype(int)
        X_df = df.drop(columns=[target_col])

        test_size = self.config.dataset.get("test_size", 0.20)
        val_size = self.config.dataset.get("val_size", 0.15)
        seed = self.config.dataset.get("random_state", 42)

        # First split: (Train + Val) and Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_df, y_binary, test_size=test_size, random_state=seed, stratify=y_binary
        )

        # Second split: Train and Val
        val_ratio_adjusted = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=seed, stratify=y_train_val
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(
        self,
        df_features: pd.DataFrame,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full training workflow.
        Returns dictionary containing model, pipeline, and split data for evaluation.
        """
        model_type = model_type or self.config.default_model_type

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data_splits(df_features)

        # Fit feature pipeline on Train fold only
        pipeline = FeaturePipeline(self.config)
        X_train_trans = pipeline.fit_transform(X_train)
        X_val_trans = pipeline.transform(X_val)
        X_test_trans = pipeline.transform(X_test)

        if model_type == "logistic_regression":
            lr_cfg = self.config.training.get("logistic_regression", {})
            model = LogisticRegression(
                max_iter=lr_cfg.get("max_iter", 1000),
                C=lr_cfg.get("C", 1.0),
                class_weight=lr_cfg.get("class_weight", "balanced"),
                solver=lr_cfg.get("solver", "lbfgs"),
                random_state=self.config.dataset.get("random_state", 42),
            )
            model.fit(X_train_trans, y_train)

        elif model_type == "lightgbm":
            lgb_cfg = self.config.training.get("lightgbm", {})
            model = lgb.LGBMClassifier(
                n_estimators=lgb_cfg.get("n_estimators", 150),
                learning_rate=lgb_cfg.get("learning_rate", 0.05),
                num_leaves=lgb_cfg.get("num_leaves", 31),
                class_weight=lgb_cfg.get("class_weight", "balanced"),
                random_state=lgb_cfg.get("random_state", 42),
                importance_type=lgb_cfg.get("importance_type", "gain"),
                verbosity=-1,
            )
            model.fit(X_train_trans, y_train)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return {
            "model": model,
            "pipeline": pipeline,
            "model_type": model_type,
            "data_splits": {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "X_train_trans": X_train_trans,
                "X_val_trans": X_val_trans,
                "X_test_trans": X_test_trans,
            },
        }
