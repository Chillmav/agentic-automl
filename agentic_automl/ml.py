"""Trusted sklearn pipelines and evaluation; no agent dependencies."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Fit every learned preprocessing operation inside the training fold."""
    models = {
        "dummy": DummyClassifier(strategy="prior", random_state=random_state),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name!r}")
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=32)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=[np.number, "bool"])),
            ("boolean", "passthrough", make_column_selector(dtype_include="bool")),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", models[model_name])])


def evaluate(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Score class 1 probabilities; classification metrics use the model's default threshold."""
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = pipeline.predict(features)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }
