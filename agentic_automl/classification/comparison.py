import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from agentic_automl.classification.data import stratified_split
from agentic_automl.classification.models.decision_tree import train_decision_tree
from agentic_automl.classification.models.knn import train_knn
from agentic_automl.classification.models.logistic_regression import (
    train_logistic_regression,
)
from agentic_automl.classification.models.random_forest import train_random_forest
from agentic_automl.classification.models.svm import train_svm
from agentic_automl.classification.validation import cross_validation_scores


def compare_classification_models(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
    test_size: float = 0.2,
    *,
    positive_class: str | int,
) -> pd.DataFrame:
    """Compare with 3-fold development CV and a separate 20% test set by default."""
    if positive_class not in target.unique():
        raise ValueError(
            f"Positive class {positive_class!r} was not found in the target."
        )
    train_x, test_x, train_y, test_y = stratified_split(
        features, target, test_size=test_size
    )
    if train_y.value_counts().min() < 3:
        raise ValueError("Each training class needs at least 3 rows for 3-fold CV.")
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(
            train_x, train_y
        )
    )
    if min(len(training) for training, _ in folds) < 5:
        raise ValueError("Each CV training fold needs at least 5 rows for KNN.")
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
        "decision_tree": train_decision_tree(
            train_x, train_y, categorical_columns=categorical_columns
        ),
        "random_forest": train_random_forest(
            train_x, train_y, categorical_columns=categorical_columns
        ),
        "svm": train_svm(train_x, train_y, categorical_columns=categorical_columns),
    }
    results = {}
    for name, model in models.items():
        cv_scores = cross_validation_scores(
            model, train_x, train_y, positive_class, folds
        )
        predictions = model.predict(test_x)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y,
            predictions,
            average="binary",
            pos_label=positive_class,
            zero_division=0,
        )
        results[name] = {
            **cv_scores,
            "test_acc": float(accuracy_score(test_y, predictions)),
            "test_recall": float(recall),
            "test_precision": float(precision),
            "test_f1": float(f1),
        }
    table = pd.DataFrame.from_dict(results, orient="index").rename_axis("model")
    table.attrs["positive_class"] = positive_class
    return table
