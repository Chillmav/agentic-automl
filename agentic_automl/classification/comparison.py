import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from agentic_automl.classification.data import stratified_split
from agentic_automl.classification.models.knn import train_knn
from agentic_automl.classification.models.logistic_regression import (
    train_logistic_regression,
)


def compare_classification_models(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
    test_size: float = 0.2,
    *,
    positive_class: str | int,
) -> pd.DataFrame:
    """Compare on one shared test split; fit both models using training rows only."""
    if positive_class not in target.unique():
        raise ValueError(
            f"Positive class {positive_class!r} was not found in the target."
        )
    train_x, test_x, train_y, test_y = stratified_split(
        features, target, test_size=test_size
    )
    models = {
        "guessing_baseline": DummyClassifier(
            strategy="stratified", random_state=42
        ).fit(train_x, train_y),
        "majority_baseline": DummyClassifier(strategy="most_frequent").fit(
            train_x, train_y
        ),
        "logistic_regression": train_logistic_regression(
            train_x, train_y, categorical_columns=categorical_columns
        ),
        "knn": train_knn(train_x, train_y, categorical_columns=categorical_columns),
    }
    results = {}
    for name, model in models.items():
        predictions = model.predict(test_x)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y,
            predictions,
            average="binary",
            pos_label=positive_class,
            zero_division=0,
        )
        results[name] = {
            "acc": float(accuracy_score(test_y, predictions)),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
        }
    table = pd.DataFrame.from_dict(results, orient="index").rename_axis("model")
    table.attrs["positive_class"] = positive_class
    return table
