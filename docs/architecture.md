# Initial architecture decisions

## Scope

Research first, CSV tabular data, independent observations, binary classification,
explicit positive class. External LLMs may receive aggregate summaries in later
stages. Local SQLite is acceptable when persistent jobs become necessary.

## Keep the first implementation small

`data.py` owns CSV validation and target encoding. `ml.py` owns trusted sklearn
pipelines and scoring. `experiment.py` owns splitting, candidate comparison, and
artifact export. `__main__.py` is a thin CLI. There is no LLM or orchestration
dependency in the training engine.

Use local JSON and joblib artifacts now. Add SQLite when jobs need persistence,
MLflow when experiment search/tracking is introduced, and FastAPI after the core
loop works. PostgreSQL and remote artifacts are deployment concerns. Regression
will need explicit task-specific estimators, scoring, and split policies; do not
generalize the binary classifier prematurely.

## Evaluation boundary

Create holdout before fitting preprocessing. Compare dummy and logistic regression
on identical development folds; evaluate only the selected model on holdout.
Mean CV ROC-AUC selects the winner; additional metrics diagnose imbalance and
threshold behavior. The classification threshold is not tuned. There is no
automatic temporal/group detection, leakage detection, or profiling-driven
feature selection. Researchers must declare independent observations and remove
leaky features. Repeated manual decisions based on holdout invalidate its role
as an untouched estimate.

## Reproducibility and artifacts

Default random state is 42. Pin dependencies and record the hash of the exact CSV
bytes parsed, split indices, CV folds, pipeline configuration, and versions.
Reserve a new artifact directory without overwriting results. The joblib pipeline
is the final development-set fit, not a full-dataset refit. Loading untrusted
serialized models is outside the supported workflow.

## Budget progression

The deterministic first step has no LLM charges. The proposed initial agent budget
is estimated USD 0.05/run rather than USD 1/run. A later planner must count token
usage, reserve estimated request cost before calls, bound retries/output tokens,
and stop or fall back when budget is unavailable. Provider billing can differ
from estimates. Trial and wall-clock limits also belong to that later executor;
the current CLI does not promise hard runtime or memory limits.
