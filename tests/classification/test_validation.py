import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from agentic_automl.classification.models.logistic_regression import (
    train_logistic_regression,
)
from agentic_automl.classification.validation import cross_validation_scores
from agentic_automl.preprocessing import TabularPreprocessor


def test_cv_refits_preprocessing_per_fold_without_changing_original_model(
    monkeypatch,
) -> None:
    features = pd.DataFrame(
        {"value": range(30), "identifier": [f"id-{i}" for i in range(30)]}
    )
    target = pd.Series(["no"] * 15 + ["yes"] * 15)
    model = train_logistic_regression(features, target)
    folds = list(
        StratifiedKFold(3, shuffle=True, random_state=42).split(features, target)
    )
    original_fit = TabularPreprocessor.fit
    fitted_rows = []

    def record_fit(self, fold_features, fold_target=None):
        result = original_fit(self, fold_features, fold_target)
        fitted_rows.append(fold_features.index.tolist())
        # Each fold has 20 unique identifiers, so the column is accepted there.
        assert self.rejected_columns_ == {}
        numeric = self.transformer_.named_transformers_["numeric"]
        assert (
            numeric.named_steps["imputer"].statistics_[0]
            == fold_features["value"].median()
        )
        return result

    monkeypatch.setattr(TabularPreprocessor, "fit", record_fit)
    scores = cross_validation_scores(model, features, target, "yes", folds)
    assert fitted_rows == [
        features.iloc[training].index.tolist() for training, _ in folds
    ]
    assert model.named_steps["preprocessor"].rejected_columns_ == {"identifier": 30}
    assert len(scores) == 8
    assert np.isfinite(list(scores.values())).all()
