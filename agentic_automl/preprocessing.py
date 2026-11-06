from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _normalize_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    return features.astype(object).where(features.notna(), np.nan)


class TabularPreprocessor(TransformerMixin, BaseEstimator):
    """Learn column selection and encoding from training data only."""

    def __init__(self, categorical_columns: list[str] | None = None) -> None:
        self.categorical_columns = categorical_columns

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> Self:
        overrides = self.categorical_columns or []
        missing = [column for column in overrides if column not in features.columns]
        if missing:
            raise ValueError(f"Categorical columns were not found: {missing}")
        numeric = [
            column
            for column in features.select_dtypes(include="number").columns
            if column not in overrides
        ]
        categorical = [column for column in features.columns if column not in numeric]
        counts = features[categorical].nunique(dropna=True)
        self.rejected_columns_ = counts[counts > 20].astype(int).to_dict()
        accepted = [
            column for column in categorical if column not in self.rejected_columns_
        ]
        if not numeric and not accepted:
            raise ValueError(
                f"No usable features remain; rejected columns: {self.rejected_columns_}"
            )

        self.transformer_ = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median", keep_empty_features=True
                                ),
                            ),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "normalize",
                                FunctionTransformer(_normalize_missing_values),
                            ),
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent", keep_empty_features=True
                                ),
                            ),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    accepted,
                ),
            ]
        )
        self.transformer_.fit(features, target)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray | spmatrix:
        return self.transformer_.transform(features)
