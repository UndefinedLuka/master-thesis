"""Matched-cohort preparation for direct-context operational-efficiency artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from visualizations.config import (
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.experiment_outcomes import model_is_excluded
from visualizations.uncertainty import mean_intervals

DIRECT_EFFICIENCY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "db_id",
    "example_id",
    "model_id",
    "pipeline",
    "format",
    "input_tokens",
    "output_tokens",
    "time_to_first_token_s",
    "generation_time_s",
    "total_time_s",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "time_to_first_token_s",
    "generation_time_s",
    "total_time_s",
)


@dataclass(frozen=True, slots=True)
class DirectEfficiencyAnalysis:
    """Matched direct-context cohort and its format-level operational means."""

    summary: pd.DataFrame
    question_counts: Mapping[str, int]
    database_counts: Mapping[str, int]
    model_count: int
    observation_count: int


def calculate_direct_efficiency(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> DirectEfficiencyAnalysis:
    """Summarize explicitly selected context-fit records without selecting questions.

    The caller selects the shared context-fit workload across both pipelines. All six
    formats must contain the same question/model pairs; invalid or missing records raise
    rather than change that workload. Blank answers and failed runs retain recorded costs.
    """
    _require_columns(results, DIRECT_EFFICIENCY_COLUMNS)
    direct = results[
        results["dataset"].isin(DATASET_ORDER)
        & results["pipeline"].eq("direct")
        & results["format"].isin(SERIALIZATION_FORMAT_ORDER)
    ].copy()
    excluded = direct["model_id"].map(
        lambda model_id: model_is_excluded(model_id, excluded_model_markers)
    )
    direct = direct.loc[~excluded]
    if direct.empty:
        raise ValueError("No retained direct-context records are available")

    data = direct
    validate_operational_pairing(data, method_column="format", methods=SERIALIZATION_FORMAT_ORDER)

    for column in NUMERIC_COLUMNS:
        if data[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"Direct operational metric {column} cannot be boolean")
        data[column] = pd.to_numeric(data[column], errors="coerce")
    missing_metrics = ~np.isfinite(data[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    if missing_metrics.any():
        raise ValueError(
            f"{int(missing_metrics.sum())} context-fit row(s) have missing or non-finite metrics"
        )
    negative_metrics = data[list(NUMERIC_COLUMNS)].lt(0).any(axis=1)
    if negative_metrics.any():
        count = int(negative_metrics.sum())
        raise ValueError(f"{count} context-fit row(s) have negative operational metrics")
    for column in ("input_tokens", "output_tokens"):
        if data[column].mod(1).ne(0).any():
            raise ValueError(f"Direct {column} must be an integer")
    if data["input_tokens"].le(0).any() or data["total_time_s"].le(0).any():
        raise ValueError("Direct input tokens and total latency must be positive")

    # Direct inference has no post-generation SQL phase, so both persisted timers should agree.
    timer_difference = (data["generation_time_s"] - data["total_time_s"]).abs()
    inconsistent_timers = timer_difference.gt(1e-6)
    if inconsistent_timers.any():
        count = int(inconsistent_timers.sum())
        raise ValueError(
            f"{count} direct row(s) have different generation and total timers"
        )
    invalid_ttft = data["time_to_first_token_s"].gt(data["total_time_s"] + 1e-6)
    if invalid_ttft.any():
        raise ValueError(
            f"{int(invalid_ttft.sum())} direct row(s) have TTFT greater than total latency"
        )

    data["generation_after_first_token_s"] = (
        data["total_time_s"] - data["time_to_first_token_s"]
    ).clip(lower=0.0)
    summary = (
        data.groupby(["dataset", "format"], as_index=False, observed=True)
        .agg(
            average_input_tokens=("input_tokens", "mean"),
            average_output_tokens=("output_tokens", "mean"),
            average_ttft_s=("time_to_first_token_s", "mean"),
            average_generation_time_s=("generation_after_first_token_s", "mean"),
            average_total_latency_s=("total_time_s", "mean"),
            observation_count=("example_id", "size"),
            question_count=("example_id", "nunique"),
            database_count=("db_id", "nunique"),
            model_count=("model_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    intervals = mean_intervals(
        data, ["dataset"], {
            "average_input_tokens": "input_tokens",
            "average_output_tokens": "output_tokens",
            "average_ttft_s": "time_to_first_token_s",
            "average_generation_time_s": "generation_after_first_token_s",
            "average_total_latency_s": "total_time_s",
        }, comparison="format", reference="csv", effect="ratio",
        pair_keys=("db_id", "example_id", "model_id"),
    )
    summary = _sort_summary(summary.merge(
        intervals, on=["dataset", "format"], validate="one_to_one"
    ))
    question_counts = {
        str(dataset): int(count)
        for dataset, count in data.groupby("dataset", observed=True)["example_id"].nunique().items()
    }
    database_counts = {
        str(dataset): int(count)
        for dataset, count in data.groupby("dataset", observed=True)["db_id"].nunique().items()
    }
    return DirectEfficiencyAnalysis(
        summary=summary,
        question_counts=question_counts,
        database_counts=database_counts,
        model_count=int(data["model_id"].nunique()),
        observation_count=len(data),
    )


def validate_operational_pairing(
    data: pd.DataFrame, *, method_column: str, methods: Sequence[str]
) -> None:
    """Validate a selected matrix, allowing synthetic model rosters but no missing cells."""
    identities = ["dataset", "example_id", "model_id", method_column]
    if data[[*identities, "db_id"]].isna().any(axis=None):
        raise ValueError("Operational records have missing run identities")
    if data.duplicated(identities).any():
        raise ValueError("Duplicate operational run identity")
    if data.groupby(["dataset", "example_id"])["db_id"].nunique().ne(1).any():
        raise ValueError("Operational questions have inconsistent database identities")
    models = set(data["model_id"])
    for dataset, rows in data.groupby("dataset", observed=True):
        expected = {
            (question, model)
            for question in rows["example_id"].unique()
            for model in models
        }
        for method in methods:
            actual = set(rows.loc[rows[method_column].eq(method),
                                  ["example_id", "model_id"]].itertuples(index=False, name=None))
            if actual != expected:
                raise ValueError(
                    f"Selected operational cohort must have identical question/model sets "
                    f"across all methods: {dataset}/{method} has {len(actual)} of "
                    f"{len(expected)} expected pairs"
                )


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    format_rank = {
        value: index for index, value in enumerate(SERIALIZATION_FORMAT_ORDER)
    }
    data = summary.copy()
    data["_dataset_rank"] = data["dataset"].map(dataset_rank)
    data["_format_rank"] = data["format"].map(format_rank)
    return data.sort_values(["_dataset_rank", "_format_rank"]).drop(
        columns=["_dataset_rank", "_format_rank"]
    ).reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(
            f"Direct operational data is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )
