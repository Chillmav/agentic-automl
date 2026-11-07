import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from agentic_automl.classification.data import stratified_split
from agentic_automl.classification.models.decision_tree import build_decision_tree
from agentic_automl.classification.models.knn import build_knn
from agentic_automl.classification.models.logistic_regression import (
    build_logistic_regression,
)
from agentic_automl.classification.models.random_forest import build_random_forest
from agentic_automl.classification.models.svm import build_svm
from agentic_automl.classification.validation import cross_validation_scores


def compare_classification_models(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
    test_size: float = 0.2,
    *,
    positive_class: str | int,
) -> pd.DataFrame:
    """Rank by mean CV F1, then evaluate only the winner on the separate test set."""
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
        "guessing_baseline": DummyClassifier(strategy="stratified", random_state=42),
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": build_logistic_regression(
            categorical_columns=categorical_columns
        ),
        "knn": build_knn(categorical_columns=categorical_columns),
        "decision_tree": build_decision_tree(categorical_columns=categorical_columns),
        "random_forest": build_random_forest(categorical_columns=categorical_columns),
        "svm": build_svm(categorical_columns=categorical_columns),
    }
    results = {}
    for name, model in models.items():
        results[name] = cross_validation_scores(
            model, train_x, train_y, positive_class, folds
        )
    # max keeps the first model in comparison order when CV scores tie.
    winner = max(results, key=lambda name: results[name]["cv_f1_mean"])
    models[winner].fit(train_x, train_y)
    predictions = models[winner].predict(test_x)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, predictions, average="binary", pos_label=positive_class, zero_division=0
    )
    table = pd.DataFrame.from_dict(results, orient="index").rename_axis("model")
    for metric, score in {
        "test_acc": accuracy_score(test_y, predictions),
        "test_recall": recall,
        "test_precision": precision,
        "test_f1": f1,
    }.items():
        table[metric] = float("nan")
        table.loc[winner, metric] = float(score)
    table.attrs.update(
        positive_class=positive_class,
        selected_model=winner,
        selection_metric="cv_f1_mean",
    )
    return table
