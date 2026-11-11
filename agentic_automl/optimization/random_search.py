from collections.abc import Callable
from typing import Literal

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline


def random_search(
    pipeline: Pipeline,
    search_space: dict[str, list],
    evaluate: Callable[[Pipeline], dict[str, float]],
    selection_metric: str,
    direction: Literal["maximize", "minimize"],
    n_trials: int = 5,
    default_scores: dict[str, float] | None = None,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    """Sample parameters and rank CV scores; never fit a final model or access test data."""
    if n_trials < 1:
        raise ValueError("Random search requires at least 1 trial.")
    if direction not in ("maximize", "minimize"):
        raise ValueError("Search direction must be maximize or minimize.")
    budget = min(n_trials, len(ParameterGrid(search_space)))
    trials = []
    defaults = pipeline.get_params()
    for parameters in ParameterSampler(search_space, n_iter=budget, random_state=42):
        if default_scores is not None and all(
            defaults[key] == value for key, value in parameters.items()
        ):
            scores = default_scores
        else:
            scores = evaluate(clone(pipeline).set_params(**parameters))
        trials.append({"parameters": parameters, **scores})
    table = pd.DataFrame(trials).rename_axis("trial")
    best_index = (
        table[selection_metric].idxmax()
        if direction == "maximize"
        else table[selection_metric].idxmin()
    )
    best = table.loc[best_index]
    winner = clone(pipeline).set_params(**best["parameters"])
    return winner, best.drop("parameters").astype(float).to_dict(), table
