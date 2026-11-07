import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from agentic_automl.regression.data import validate_regression_target
from agentic_automl.regression.models.decision_tree import build_decision_tree
from agentic_automl.regression.models.gradient_boosting import build_gradient_boosting
from agentic_automl.regression.models.linear_regression import build_linear_regression
from agentic_automl.regression.models.random_forest import build_random_forest
from agentic_automl.regression.models.ridge import build_ridge
from agentic_automl.regression.validation import (
    cross_validation_scores,
    regression_cv_folds,
)


def compare_regression_models(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
    test_size: float | int = 0.2,
    shuffle: bool = True,
    dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Select by lowest mean CV MSE; test only the winner. Ordered dates stay together."""
    validate_regression_target(target)
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    if dates is not None:
        if shuffle or not dates.index.equals(features.index):
            raise ValueError("Dates require matching indices and shuffle=False.")
        if dates.isna().any() or not dates.is_monotonic_increasing:
            raise ValueError(
                "Dates must be non-missing and sorted from earliest to latest."
            )
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=test_size, random_state=42, shuffle=shuffle
    )
    if len(test_y) < 2:
        raise ValueError("At least two test rows are required to calculate R2.")
    train_dates = dates.iloc[: len(train_x)] if dates is not None else None
    if dates is not None and train_dates.max() >= dates.iloc[len(train_x) :].min():
        raise ValueError(
            "The final test split must keep whole dates together; adjust test_size."
        )
    folds = regression_cv_folds(train_x, shuffle=shuffle, dates=train_dates)
    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": build_linear_regression(
            categorical_columns=categorical_columns
        ),
        "ridge": build_ridge(categorical_columns=categorical_columns),
        "decision_tree": build_decision_tree(categorical_columns=categorical_columns),
        "random_forest": build_random_forest(categorical_columns=categorical_columns),
        "gradient_boosting": build_gradient_boosting(
            categorical_columns=categorical_columns
        ),
    }
    results = {}
    for name, model in models.items():
        results[name] = cross_validation_scores(model, train_x, train_y, folds)
    winner = min(results, key=lambda name: results[name]["cv_mse_mean"])
    models[winner].fit(train_x, train_y)
    predictions = models[winner].predict(test_x)
    table = pd.DataFrame.from_dict(results, orient="index").rename_axis("model")
    for metric, score in {
        "test_mse": mean_squared_error(test_y, predictions),
        "test_r2": r2_score(test_y, predictions),
    }.items():
        table[metric] = float("nan")
        table.loc[winner, metric] = float(score)
    table.attrs.update(selected_model=winner, selection_metric="cv_mse_mean")
    table.attrs["rejected_columns"] = (
        models[winner].named_steps["preprocessor"].rejected_columns_
        if hasattr(models[winner], "named_steps")
        else {}
    )
    return table
