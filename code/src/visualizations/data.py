"""Shared loading and filtering utilities for result visualizations.

Scored result trees can be large because every record contains the model's full response.  The
loader therefore reads NDJSON incrementally and lets each visualization select only the columns
it needs. Nested fields use dot-separated names, for example
``scores.normalized_exact_match``. Paths also traverse lists, so
``attempts.llm_metrics.time_to_first_token_s`` returns one value per retry attempt.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from visualizations.config import SQL_EXECUTION_LABEL

SCORED_SUFFIX = ".scored.ndjson"
SOURCE_FILE_COLUMN = "source_file"
HAS_ANSWER_COLUMN = "has_answer"


@dataclass(frozen=True, slots=True)
class CompleteQuestionCohort:
    """Successful rows from questions completed by every selected model and method."""

    records: pd.DataFrame
    question_count: int
    observation_count: int

# Lightweight fields that are useful to most analyses. Large response/attempt payloads are
# intentionally opt-in through the ``columns`` argument to ``load_scored_results``.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "example_id",
    "db_id",
    "dataset",
    "pipeline",
    "format",
    "model_id",
    "input_tokens",
    "output_tokens",
    "time_to_first_token_s",
    "reasoning_time_s",
    "generation_time_s",
    "query_execution_time_s",
    "total_time_s",
    "retry_count",
    "execution_success",
    "definitive_failure",
    "error",
    HAS_ANSWER_COLUMN,
    "scores.normalized_exact_match",
    "scores.rouge_l_f",
    "scores.bertscore_p",
    "scores.bertscore_r",
    "scores.bertscore_f",
    "execution.execution_accuracy",
    "execution.result_match",
    SOURCE_FILE_COLUMN,
)


def discover_scored_files(target: Path) -> list[Path]:
    """Return scored NDJSON files beneath ``target`` in deterministic order.

    ``target`` may be either one ``*.scored.ndjson`` file or a directory containing a result
    tree. Unscored sibling files are deliberately ignored to avoid counting each run twice.
    """
    target = target.expanduser()
    if target.is_file():
        if not target.name.endswith(SCORED_SUFFIX):
            raise ValueError(f"Expected a {SCORED_SUFFIX} file, got: {target}")
        return [target]
    if target.is_dir():
        return sorted(path for path in target.rglob(f"*{SCORED_SUFFIX}") if path.is_file())
    raise FileNotFoundError(f"Results path does not exist: {target}")


def select_complete_question_cohort(results: pd.DataFrame) -> CompleteQuestionCohort:
    """Select a common question cohort for unbiased cross-method efficiency means.

    A question is retained within a dataset only when every model--method combination present
    in that dataset has a row with no pipeline error and a non-empty generated answer. The
    returned records therefore give every displayed method the same questions and models.
    Text-to-SQL success here means successful generation, matching the historical filtering of
    the efficiency plots; SQL execution correctness is analyzed separately.
    """
    required_columns = {
        "dataset",
        "example_id",
        "model_id",
        "pipeline",
        "format",
        "error",
        HAS_ANSWER_COLUMN,
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing complete-cohort column(s): {names}")
    if results.empty:
        return CompleteQuestionCohort(results.copy(), 0, 0)

    data = results.copy()
    method_column = "_complete_cohort_method"
    data[method_column] = data["format"]
    is_sql = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql, method_column] = SQL_EXECUTION_LABEL
    valid_identity = data[["dataset", "example_id", "model_id", method_column]].notna().all(axis=1)
    data = data.loc[valid_identity].copy()

    successful = data["error"].isna() & data[HAS_ANSWER_COLUMN].eq(True)
    eligible_keys: list[tuple[object, object]] = []
    for dataset, dataset_rows in data.groupby("dataset", observed=True, sort=False):
        models = dataset_rows["model_id"].drop_duplicates()
        methods = dataset_rows[method_column].drop_duplicates()
        expected_count = len(models) * len(methods)
        if expected_count == 0:
            continue

        successful_keys = dataset_rows.loc[
            successful.loc[dataset_rows.index],
            ["example_id", "model_id", method_column],
        ].drop_duplicates()
        coverage = successful_keys.groupby("example_id", observed=True).size()
        eligible_keys.extend(
            (dataset, example_id)
            for example_id in coverage.index[coverage.eq(expected_count)]
        )

    if not eligible_keys:
        empty = data.iloc[0:0].drop(columns=method_column)
        return CompleteQuestionCohort(empty, 0, 0)

    eligible_index = pd.MultiIndex.from_tuples(eligible_keys, names=["dataset", "example_id"])
    row_index = pd.MultiIndex.from_frame(data[["dataset", "example_id"]])
    selected = data.loc[row_index.isin(eligible_index) & successful].drop(columns=method_column)
    return CompleteQuestionCohort(
        records=selected.copy(),
        question_count=len(eligible_keys),
        observation_count=len(selected),
    )


def load_scored_results(
    target: Path,
    *,
    columns: Sequence[str] | None = DEFAULT_COLUMNS,
    datasets: Collection[str] | None = None,
    pipelines: Collection[str] | None = None,
    formats: Collection[str] | None = None,
    models: Collection[str] | None = None,
    answers_only: bool = False,
) -> pd.DataFrame:
    """Load and filter scored result records into a tidy pandas DataFrame.

    Args:
        target: A scored NDJSON file or a directory searched recursively.
        columns: Dot-separated fields to retain. The default omits large response payloads;
            pass ``None`` to flatten and retain every field.
        datasets: Optional exact dataset-name filter.
        pipelines: Optional exact pipeline-name filter.
        formats: Optional exact serialization-format filter.
        models: Optional exact model-id filter.
        answers_only: Keep only records with no pipeline error and a non-empty ``raw_output``.
            This excludes failures such as ``context_exceeded`` whose token fields are zero even
            though no answer was generated.

    Malformed lines are skipped with a warning so an interrupted result write does not prevent
    the remaining completed records from being visualized.
    """
    selected_columns = _validated_columns(columns)
    rows: list[dict[str, Any]] = []

    for source_file in discover_scored_files(target):
        for record in _read_records(source_file):
            if not _matches_filters(
                record,
                datasets=datasets,
                pipelines=pipelines,
                formats=formats,
                models=models,
            ):
                continue
            if answers_only and (record.get("error") is not None or not record.get("raw_output")):
                continue

            if selected_columns is None:
                row = _flatten(record)
                row[SOURCE_FILE_COLUMN] = str(source_file)
            else:
                row = {
                    column: _project_value(record, column, source_file)
                    for column in selected_columns
                }
            rows.append(row)

    if selected_columns is not None:
        return pd.DataFrame.from_records(rows, columns=selected_columns)
    return pd.DataFrame.from_records(rows)


def _validated_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    selected = tuple(columns)
    if any(not column for column in selected):
        raise ValueError("Column names must be non-empty")
    if len(set(selected)) != len(selected):
        raise ValueError("Column names must be unique")
    return selected


def _read_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _warn_skipped_line(path, line_number, f"invalid JSON ({exc.msg})")
                continue
            if not isinstance(record, dict):
                _warn_skipped_line(path, line_number, "record is not a JSON object")
                continue
            yield record


def _warn_skipped_line(path: Path, line_number: int, reason: str) -> None:
    print(
        f"warning: skipping line {line_number} in {path}: {reason}",
        file=sys.stderr,
    )


def _matches_filters(
    record: Mapping[str, Any],
    *,
    datasets: Collection[str] | None,
    pipelines: Collection[str] | None,
    formats: Collection[str] | None,
    models: Collection[str] | None,
) -> bool:
    return (
        _matches(record.get("dataset"), datasets)
        and _matches(record.get("pipeline"), pipelines)
        and _matches(record.get("format"), formats)
        and _matches(record.get("model_id"), models)
    )


def _matches(value: object, allowed: Collection[str] | None) -> bool:
    return allowed is None or value in allowed


def _project_value(record: Mapping[str, Any], column: str, source_file: Path) -> Any:
    if column == SOURCE_FILE_COLUMN:
        return str(source_file)
    if column == HAS_ANSWER_COLUMN:
        return bool(record.get("raw_output"))
    return _nested_value(record, column)


def _nested_value(record: Mapping[str, Any], column: str) -> Any:
    return _descend(record, column.split("."))


def _descend(value: Any, parts: list[str]) -> Any:
    if not parts:
        return value
    if isinstance(value, Mapping):
        return _descend(value.get(parts[0]), parts[1:])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_descend(item, parts) for item in value]
    return None


def _flatten(record: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(child, child_prefix)
        else:
            flattened[prefix] = value

    visit(record, "")
    return flattened
