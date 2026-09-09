"""Shared preparation for Text-to-SQL retrieval and operational analyses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from visualizations.config import (
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    TOKENIZER_FAMILY_LABELS,
)
from visualizations.direct_accuracy_data import (
    METRIC_SOURCE_COLUMNS,
    model_size_label,
    parse_model_family_size,
)
from visualizations.experiment_outcomes import model_is_excluded

ATTEMPT_TTFT_COLUMN = "attempts.llm_metrics.time_to_first_token_s"
ATTEMPT_GENERATION_COLUMN = "attempts.llm_metrics.total_generation_time_s"
TEXT_TO_SQL_COLUMNS: tuple[str, ...] = (
    "dataset",
    "db_id",
    "example_id",
    "model_id",
    "pipeline",
    "format",
    "error",
    "input_tokens",
    "output_tokens",
    "time_to_first_token_s",
    "generation_time_s",
    "query_execution_time_s",
    "total_time_s",
    "retry_count",
    "execution_success",
    *METRIC_SOURCE_COLUMNS.values(),
    "execution.result_match",
    ATTEMPT_TTFT_COLUMN,
    ATTEMPT_GENERATION_COLUMN,
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "time_to_first_token_s",
    "generation_time_s",
    "query_execution_time_s",
    "total_time_s",
    "retry_count",
    *METRIC_SOURCE_COLUMNS.values(),
)


@dataclass(frozen=True, slots=True)
class TextToSqlAnalysis:
    """Validated Text-to-SQL records and their dataset/model summaries."""

    records: pd.DataFrame
    dataset_summary: pd.DataFrame
    model_summary: pd.DataFrame
    run_count: int
    model_count: int
    question_counts: dict[str, int]


def calculate_text_to_sql_analysis(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> TextToSqlAnalysis:
    """Validate retained Text-to-SQL runs and calculate dataset/model means."""
    _require_columns(results, TEXT_TO_SQL_COLUMNS)
    data = results[
        results["dataset"].isin(DATASET_ORDER) & results["pipeline"].eq("text_to_sql")
    ].copy()
    excluded = data["model_id"].map(
        lambda model_id: model_is_excluded(model_id, excluded_model_markers)
    )
    data = data.loc[~excluded]
    if data.empty:
        raise ValueError("No retained Text-to-SQL records are available")

    duplicate = data.duplicated(["dataset", "model_id", "example_id"], keep=False)
    if duplicate.any():
        identity = data.loc[duplicate, ["dataset", "model_id", "example_id"]].iloc[0]
        raise ValueError(f"Duplicate Text-to-SQL run identity: {identity.to_dict()}")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    missing_numeric = data[list(NUMERIC_COLUMNS)].isna().any(axis=1)
    if missing_numeric.any():
        raise ValueError(
            f"{int(missing_numeric.sum())} Text-to-SQL row(s) have missing numeric metrics"
        )
    negative_operational = data[
        [
            "input_tokens",
            "output_tokens",
            "time_to_first_token_s",
            "generation_time_s",
            "query_execution_time_s",
            "total_time_s",
            "retry_count",
        ]
    ].lt(0).any(axis=1)
    if negative_operational.any():
        raise ValueError(
            f"{int(negative_operational.sum())} Text-to-SQL row(s) have negative metrics"
        )

    timer_difference = (
        data["generation_time_s"]
        + data["query_execution_time_s"]
        - data["total_time_s"]
    ).abs()
    inconsistent = timer_difference.gt(1e-6)
    if inconsistent.any():
        raise ValueError(
            f"{int(inconsistent.sum())} Text-to-SQL row(s) have inconsistent total latency"
        )

    data["attempt_generation_times_s"] = data[ATTEMPT_GENERATION_COLUMN].map(
        _numeric_attempt_values
    )
    invalid_attempt_times = data["attempt_generation_times_s"].isna()
    if invalid_attempt_times.any():
        raise ValueError(
            f"{int(invalid_attempt_times.sum())} Text-to-SQL row(s) lack attempt generation data"
        )
    attempt_counts = data["attempt_generation_times_s"].map(len)
    mismatched_attempt_counts = attempt_counts.ne(data["retry_count"] + 1)
    if mismatched_attempt_counts.any():
        raise ValueError(
            f"{int(mismatched_attempt_counts.sum())} Text-to-SQL row(s) have inconsistent "
            "attempt counts"
        )
    attempt_generation_totals = data["attempt_generation_times_s"].map(sum)
    mismatched_generation = (attempt_generation_totals - data["generation_time_s"]).abs().gt(
        1e-6
    )
    if mismatched_generation.any():
        raise ValueError(
            f"{int(mismatched_generation.sum())} Text-to-SQL row(s) have inconsistent "
            "per-attempt generation time"
        )
    data["retry_generation_overhead_s"] = data["attempt_generation_times_s"].map(
        lambda values: sum(values[1:])
    )
    data["attempt_count"] = attempt_counts

    data["cumulative_ttft_s"] = data[ATTEMPT_TTFT_COLUMN].map(_sum_attempt_values)
    missing_attempt_timing = data["cumulative_ttft_s"].isna()
    if missing_attempt_timing.any():
        raise ValueError(
            f"{int(missing_attempt_timing.sum())} Text-to-SQL row(s) lack attempt TTFT data"
        )
    invalid_ttft = data["cumulative_ttft_s"].gt(data["generation_time_s"] + 1e-6)
    if invalid_ttft.any():
        raise ValueError(
            f"{int(invalid_ttft.sum())} Text-to-SQL row(s) have cumulative TTFT above model time"
        )
    data["generation_after_ttft_s"] = (
        data["generation_time_s"] - data["cumulative_ttft_s"]
    ).clip(lower=0.0)

    metadata = data["model_id"].map(parse_model_family_size)
    unsupported = metadata.isna()
    if unsupported.any():
        model_id = str(data.loc[unsupported, "model_id"].iloc[0])
        raise ValueError(f"Unsupported Text-to-SQL model identifier: {model_id}")
    data["model_family"] = [value[0] for value in metadata]
    data["model_size_b"] = [value[1] for value in metadata]
    data["size_label"] = data["model_id"].map(model_size_label)
    data["execution_success_numeric"] = data["execution_success"].astype(float)
    data["result_match_numeric"] = data["execution.result_match"].astype(float)

    aggregations: dict[str, tuple[str, str]] = {
        "normalized_exact_match": (
            METRIC_SOURCE_COLUMNS["normalized_exact_match"],
            "mean",
        ),
        "bertscore_f": (METRIC_SOURCE_COLUMNS["bertscore_f"], "mean"),
        "rouge_l_f": (METRIC_SOURCE_COLUMNS["rouge_l_f"], "mean"),
        "execution_success_rate": ("execution_success_numeric", "mean"),
        "result_match_rate": ("result_match_numeric", "mean"),
        "average_input_tokens": ("input_tokens", "mean"),
        "average_output_tokens": ("output_tokens", "mean"),
        "average_cumulative_ttft_s": ("cumulative_ttft_s", "mean"),
        "average_generation_time_s": ("generation_after_ttft_s", "mean"),
        "average_sql_execution_time_s": ("query_execution_time_s", "mean"),
        "average_total_latency_s": ("total_time_s", "mean"),
        "average_retries": ("retry_count", "mean"),
        "run_count": ("example_id", "size"),
        "question_count": ("example_id", "nunique"),
    }
    dataset_summary = (
        data.groupby("dataset", as_index=False, observed=True)
        .agg(**aggregations)
        .reset_index(drop=True)
    )
    dataset_summary = _sort_dataset_summary(dataset_summary)

    model_summary = (
        data.groupby(
            ["dataset", "model_id", "model_family", "model_size_b", "size_label"],
            as_index=False,
            observed=True,
        )
        .agg(**aggregations)
        .reset_index(drop=True)
        .sort_values(["dataset", "model_family", "model_size_b", "model_id"])
        .reset_index(drop=True)
    )
    question_counts = {
        str(dataset): int(count)
        for dataset, count in data.groupby("dataset", observed=True)["example_id"].nunique().items()
    }
    return TextToSqlAnalysis(
        records=data.reset_index(drop=True),
        dataset_summary=dataset_summary,
        model_summary=model_summary,
        run_count=len(data),
        model_count=int(data["model_id"].nunique()),
        question_counts=question_counts,
    )


def text_to_sql_model_label(model_family: str, size_label: str) -> str:
    """Return a compact two-line publication label for a model."""
    size_text = size_label.replace(" (", "B (", 1) if " (" in size_label else f"{size_label}B"
    return f"{TOKENIZER_FAMILY_LABELS[model_family]}\n{size_text}"


def _numeric_attempt_values(values: Any) -> list[float] | None:
    if not isinstance(values, list) or not values:
        return None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any() or numeric.lt(0).any():
        return None
    return [float(value) for value in numeric]


def _sum_attempt_values(values: Any) -> float | None:
    numeric = _numeric_attempt_values(values)
    return sum(numeric) if numeric is not None else None


def _sort_dataset_summary(summary: pd.DataFrame) -> pd.DataFrame:
    positions = {value: index for index, value in enumerate(DATASET_ORDER)}
    return summary.sort_values(
        "dataset", key=lambda values: values.map(positions)
    ).reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(
            f"Text-to-SQL data is missing required column(s): {', '.join(sorted(missing))}"
        )
