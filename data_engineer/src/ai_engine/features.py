"""
Feature Preprocessing & Transformation Pipeline.
Builds scikit-learn transformers that prevent data snooping and leakage.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ..config import ModelConfig, load_model_config


class FeaturePipeline:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.numerical_cols = self.config.features.numerical
        self.categorical_cols = self.config.features.categorical + self.config.features.binary
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []

    def build_preprocessor(self) -> ColumnTransformer:
        """Construct ColumnTransformer for numerical and categorical features."""
        num_pipeline = Pipeline([
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_cols),
                ("cat", cat_pipeline, self.categorical_cols),
            ],
            remainder="drop",
        )
        return self.preprocessor

    def fit(self, X: pd.DataFrame) -> "FeaturePipeline":
        """Fit preprocessor only on Training fold."""
        if self.preprocessor is None:
            self.build_preprocessor()

        self.preprocessor.fit(X)

        # Extract output feature names after transformation
        encoded_cat_names = (
            self.preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(self.categorical_cols)
            .tolist()
        )
        self.feature_names = self.numerical_cols + encoded_cat_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features into model-ready array."""
        if self.preprocessor is None:
            raise ValueError("FeaturePipeline has not been fitted yet!")
        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)
