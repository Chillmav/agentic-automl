"""A small LangGraph workflow: prepare saved context, then request advice."""

import json
from collections.abc import Callable
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_automl.agents.ollama import review_with_ollama


class ColumnExclusion(BaseModel):
    kind: Literal["exclude_column"]
    column: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=1200)


class Analysis(BaseModel):
    summary: str = Field(min_length=1, max_length=2400)
    findings: list[str] = Field(max_length=5)
    next_steps: list[str] = Field(min_length=1, max_length=3)
    proposed_action: ColumnExclusion | None = None


class AnalystState(TypedDict, total=False):
    experiment: dict
    context: str
    analysis: dict


def prepare_context(state: AnalystState) -> dict:
    record = state["experiment"]
    context = {
        key: record[key]
        for key in (
            "task",
            "target_column",
            "positive_class",
            "sample_rows",
            "model_names",
            "selected_model",
            "selection_metric",
            "cv_results",
            "rejected_columns",
            "dataset_review",
            "training_profile",
        )
        if key in record
    }
    # Full-upload summaries are retained in history; prefer training evidence here.
    if "training_profile" in context and "dataset_review" in context:
        context["dataset_review"] = {
            key: value
            for key, value in context["dataset_review"].items()
            if key != "columns"
        }
    serialized = json.dumps(context, allow_nan=False)
    if len(serialized) > 20_000:
        raise RuntimeError(
            "This experiment summary exceeds the small analyst's 20,000-character limit."
        )
    return {"context": serialized}


def analyze_experiment(
    experiment: dict,
    reviewer: Callable[[str], str] | None = None,
) -> dict:
    """Run exactly one model call. Never train, modify data or write history."""
    reviewer = reviewer or review_with_ollama

    def recommend(state: AnalystState) -> dict:
        analysis = Analysis.model_validate_json(reviewer(state["context"]))
        if analysis.proposed_action:
            columns = experiment.get("training_profile", {}).get("columns", [])
            allowed = {column["name"] for column in columns}
            if len(allowed) < 2 or analysis.proposed_action.column not in allowed:
                raise ValueError(
                    "The suggested column is not an eligible training feature."
                )
        return {"analysis": analysis.model_dump()}

    graph = StateGraph(AnalystState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("recommend", recommend)
    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile().invoke({"experiment": experiment})["analysis"]
