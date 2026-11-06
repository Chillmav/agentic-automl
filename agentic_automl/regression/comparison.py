import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from agentic_automl.regression.data import validate_regression_target
from agentic_automl.regression.models.linear_regression import train_linear_regression


def compare_regression_models(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str] | None = None,
    test_size: float | int = 0.2,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Return MSE/R2; shuffle=False reserves the final rows for testing."""
    validate_regression_target(target)
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have matching row indices.")
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=test_size, random_state=42, shuffle=shuffle
    )
    if len(test_y) < 2:
        raise ValueError("At least two test rows are required to calculate R2.")
    models = {
        "mean_baseline": DummyRegressor(strategy="mean").fit(train_x, train_y),
        "linear_regression": train_linear_regression(
            train_x, train_y, categorical_columns
        ),
    }
    results = {}
    for name, model in models.items():
        predictions = model.predict(test_x)
        results[name] = {
            "mse": float(mean_squared_error(test_y, predictions)),
            "r2": float(r2_score(test_y, predictions)),
        }
    table = pd.DataFrame.from_dict(results, orient="index").rename_axis("model")
    table.attrs["rejected_columns"] = (
        models["linear_regression"].named_steps["preprocessor"].rejected_columns_
    )
    return table
