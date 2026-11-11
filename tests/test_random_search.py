import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from agentic_automl.optimization.random_search import random_search
from agentic_automl.regression.models.ridge import build_ridge


@pytest.mark.parametrize(
    ("direction", "best_index"), [("minimize", 0), ("maximize", 1)]
)
def test_shared_search_direction_reproducibility_and_unfitted_result(
    direction, best_index
) -> None:
    original = build_ridge()
    tables = []
    for _ in range(2):
        values = iter([0.2, 0.8])

        def evaluate(model):
            with pytest.raises(NotFittedError):
                check_is_fitted(model)
            return {"score": next(values)}

        model, scores, table = random_search(
            original,
            {"estimator__alpha": [0.1, 10.0]},
            evaluate,
            "score",
            direction,
            n_trials=10,
        )
        assert len(table) == 2
        assert scores["score"] == table.loc[best_index, "score"]
        assert (
            model.get_params()["estimator__alpha"]
            == table.loc[best_index, "parameters"]["estimator__alpha"]
        )
        assert original.get_params()["estimator__alpha"] == 1.0
        with pytest.raises(NotFittedError):
            check_is_fitted(model)
        tables.append(table)
    pd.testing.assert_frame_equal(*tables)


def test_invalid_search_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction"):
        random_search(
            build_ridge(),
            {"estimator__alpha": [1]},
            lambda _: {"score": 1},
            "score",
            "unknown",
        )
