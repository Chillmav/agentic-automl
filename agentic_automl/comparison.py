import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from agentic_automl.data import stratified_split
from agentic_automl.model import train_logistic_regression


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
    test_size: float = 0.2,
) -> dict[str, dict[str, float]]:
    """Compare on one shared test split; fit both models using training rows only."""
    train_x, test_x, train_y, test_y = stratified_split(
        features, target, test_size=test_size
    )
    models = {
        "guessing_baseline": DummyClassifier(
            strategy="stratified", random_state=42
        ).fit(train_x, train_y),
        "logistic_regression": train_logistic_regression(
            train_x, train_y, categorical_columns=categorical_columns
        ),
    }
    results = {}
    for name, model in models.items():
        predictions = model.predict(test_x)
        results[name] = {
            "accuracy": float(accuracy_score(test_y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        }
    return results
