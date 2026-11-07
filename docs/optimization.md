# Optimization boundaries

Model builders create unfitted sklearn pipelines. Task validation functions evaluate
them on fixed development folds, including fold-local preprocessing. Classification
maximizes mean CV F1; regression minimizes mean CV MSE.

`optimization/random_search.py` owns seeded parameter sampling, cloning, default-score
reuse, trial records, and ranking. It receives an evaluation callback and metric
direction, with no knowledge of datasets, test sets, or agent orchestration.
Task-specific `random_search.py` modules own parameter spaces and CV adapters.

Future Optuna tuning can reuse the builders, task evaluators, and development folds.
A cheap random-search screen followed by deeper tuning of its winner is a practical
budget policy, not a guarantee of a global optimum. It may miss families whose good
configurations were not sampled; optionally retain a small shortlist later.

Both optimization stages must finish before fitting the final model on all
development rows and evaluating the test set once. The current runner stops after
random search; Optuna, agentic planning, and persistence are not implemented.
