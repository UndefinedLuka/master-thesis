"""Generate the Chapter 5 LaTeX table of terminal experiment outcomes.

Every scored benchmark record is assigned to exactly one outcome: successful on the first
attempt, successful after Text-to-SQL correction, or one of the terminal failure categories.
The Qwen 3.5 27B records are excluded by default because their thinking and visible-output
tokens cannot be separated reliably.

The module contains calculation, rendering, and one-table generation logic only. The sole
visualization command in :mod:`visualizations.generate_all` loads and validates the result
snapshot before invoking this generator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from serialization.types import FORMATS
from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    EXPECTED_QUESTION_COUNTS,
    EXPERIMENT_OUTCOMES_TABLE_FILENAME,
    EXPERIMENT_OUTCOMES_TABLE_LABEL,
    LATEX_ROW_END,
    OUTCOME_LABELS,
)
from visualizations.latex_numbers import num

DEFAULT_DATASETS = DATASET_ORDER
EXPECTED_RETAINED_MODEL_COUNT = 10
SQL_METHOD = "text_to_sql"
EXPECTED_METHODS = frozenset((*FORMATS, SQL_METHOD))

SUCCESS_CLEAN = "success_clean"
SUCCESS_AFTER_CORRECTION = "success_after_correction"
FAILED_CONTEXT = "failed_context"
FAILED_SQL = "failed_sql"
FAILED_NO_SQL = "failed_no_sql"
FAILED_GENERATION = "failed_generation"
FAILED_DIRECT_OUTPUT = "failed_direct_output"
FAILED_OTHER = "failed_other"

SUCCESS_OUTCOMES: tuple[str, ...] = (SUCCESS_CLEAN, SUCCESS_AFTER_CORRECTION)
FAILURE_OUTCOMES: tuple[str, ...] = (
    FAILED_CONTEXT,
    FAILED_SQL,
    FAILED_NO_SQL,
    FAILED_GENERATION,
    FAILED_DIRECT_OUTPUT,
    FAILED_OTHER,
)
OUTCOME_ORDER: tuple[str, ...] = (*SUCCESS_OUTCOMES, *FAILURE_OUTCOMES)

OUTCOME_COLUMNS: tuple[str, ...] = (
    "dataset",
    "pipeline",
    "format",
    "model_id",
    "example_id",
    "predicted_sql",
    "retry_count",
    "execution_success",
    "definitive_failure",
    "error",
    "has_answer",
)


@dataclass(frozen=True, slots=True)
class CoverageStats:
    """Dimensions of a result snapshot that passed complete-coverage validation."""

    question_counts: Mapping[str, int]
    model_count: int
    run_count: int


@dataclass(frozen=True, slots=True)
class OutcomeSummary:
    """Terminal outcome counts after applying the configured model exclusion."""

    counts: Mapping[str, Mapping[str, int]]
    datasets: tuple[str, ...]
    retained_run_count: int
    excluded_run_count: int
    retained_model_count: int

    def count(self, outcome: str, dataset: str) -> int:
        """Return one outcome/dataset count, treating an absent cell as zero."""
        return int(self.counts.get(outcome, {}).get(dataset, 0))

    def outcome_total(self, outcome: str) -> int:
        """Return an outcome count summed across displayed datasets."""
        return sum(self.count(outcome, dataset) for dataset in self.datasets)



def classify_outcome(record: Mapping[str, Any]) -> str:
    """Assign one result record to exactly one terminal outcome category."""
    error = _text(record.get("error"))
    if error == "context_exceeded":
        return FAILED_CONTEXT
    if error:
        return FAILED_GENERATION

    pipeline = _text(record.get("pipeline"))
    if pipeline == "direct":
        return SUCCESS_CLEAN if _is_true(record.get("has_answer")) else FAILED_DIRECT_OUTPUT

    if pipeline != SQL_METHOD:
        return FAILED_OTHER

    execution_success = record.get("execution_success")
    if _is_true(execution_success):
        retry_count = _nonnegative_int(record.get("retry_count"))
        return SUCCESS_AFTER_CORRECTION if retry_count > 0 else SUCCESS_CLEAN

    if _is_false(execution_success):
        predicted_sql = _text(record.get("predicted_sql"))
        return FAILED_SQL if predicted_sql else FAILED_NO_SQL

    return FAILED_OTHER


def calculate_outcome_summary(
    results: pd.DataFrame,
    *,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> OutcomeSummary:
    """Exclude configured models, classify retained records, and aggregate their counts."""
    _require_columns(results, OUTCOME_COLUMNS)
    selected_datasets = tuple(datasets)
    selected = results[results["dataset"].isin(selected_datasets)].copy()
    excluded_mask = selected["model_id"].map(
        lambda model_id: model_is_excluded(model_id, excluded_model_markers)
    )
    retained = selected.loc[~excluded_mask]
    excluded_run_count = int(excluded_mask.sum())

    counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in retained.to_dict("records"):
        counts[classify_outcome(record)][_text(record["dataset"])] += 1

    normalized_counts = {
        outcome: {dataset: counts[outcome][dataset] for dataset in selected_datasets}
        for outcome in OUTCOME_ORDER
    }
    retained_models = {_text(model) for model in retained["model_id"].dropna().unique()}
    return OutcomeSummary(
        counts=normalized_counts,
        datasets=selected_datasets,
        retained_run_count=len(retained),
        excluded_run_count=excluded_run_count,
        retained_model_count=len(retained_models),
    )


def validate_complete_coverage(
    results: pd.DataFrame,
    *,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
    expected_question_counts: Mapping[str, int] = EXPECTED_QUESTION_COUNTS,
    expected_model_count: int = EXPECTED_RETAINED_MODEL_COUNT,
) -> CoverageStats:
    """Validate the complete model × question × method matrix used by Chapter 5.

    Validation happens after model exclusion. It rejects duplicate run identities, unexpected
    pipeline/format combinations, missing datasets or models, incorrect question counts, and
    any model/question pair that does not contain all six direct formats plus Text-to-SQL.
    """
    _require_columns(results, ("dataset", "pipeline", "format", "model_id", "example_id"))
    selected_datasets = tuple(datasets)
    selected = results[results["dataset"].isin(selected_datasets)].copy()
    excluded_mask = selected["model_id"].map(
        lambda model_id: model_is_excluded(model_id, excluded_model_markers)
    )
    retained = selected.loc[~excluded_mask].copy()
    if retained.empty:
        raise ValueError("No retained experiment records are available")

    retained["_method"] = retained.apply(_method_for_record, axis=1)
    identity_columns = ["dataset", "model_id", "example_id", "_method"]
    duplicate_mask = retained.duplicated(identity_columns, keep=False)
    if duplicate_mask.any():
        duplicate = retained.loc[duplicate_mask, identity_columns].iloc[0].to_dict()
        raise ValueError(f"Duplicate experiment run identity: {duplicate}")

    global_models = {_text(model) for model in retained["model_id"].dropna().unique()}
    if len(global_models) != expected_model_count:
        raise ValueError(
            f"Expected {expected_model_count} retained models, found {len(global_models)}"
        )

    question_counts: dict[str, int] = {}
    for dataset in selected_datasets:
        dataset_rows = retained[retained["dataset"].eq(dataset)]
        if dataset_rows.empty:
            raise ValueError(f"No retained records found for dataset {dataset!r}")

        dataset_models = {
            _text(model) for model in dataset_rows["model_id"].dropna().unique()
        }
        if dataset_models != global_models:
            missing = sorted(global_models.difference(dataset_models))
            extra = sorted(dataset_models.difference(global_models))
            raise ValueError(
                f"Model coverage differs for {dataset}: missing={missing}, extra={extra}"
            )

        questions = {
            _text(example_id) for example_id in dataset_rows["example_id"].dropna().unique()
        }
        expected_questions = expected_question_counts.get(dataset)
        if expected_questions is not None and len(questions) != expected_questions:
            raise ValueError(
                f"Expected {expected_questions:,} questions for {dataset}, "
                f"found {len(questions):,}"
            )
        question_counts[dataset] = len(questions)

        method_sets = dataset_rows.groupby(
            ["model_id", "example_id"], observed=True
        )["_method"].agg(frozenset)
        expected_pair_count = len(global_models) * len(questions)
        if len(method_sets) != expected_pair_count:
            raise ValueError(
                f"Incomplete model/question coverage for {dataset}: expected "
                f"{expected_pair_count:,} pairs, found {len(method_sets):,}"
            )
        incomplete = method_sets[method_sets.map(lambda methods: methods != EXPECTED_METHODS)]
        if not incomplete.empty:
            model_id, example_id = incomplete.index[0]
            missing_methods = sorted(EXPECTED_METHODS.difference(incomplete.iloc[0]))
            raise ValueError(
                f"Incomplete method coverage for {dataset}/{model_id}/{example_id}: "
                f"missing {missing_methods}"
            )

    expected_run_count = sum(question_counts.values()) * expected_model_count * len(
        EXPECTED_METHODS
    )
    if len(retained) != expected_run_count:
        raise ValueError(
            f"Expected {expected_run_count:,} retained runs, found {len(retained):,}"
        )
    return CoverageStats(
        question_counts=question_counts,
        model_count=len(global_models),
        run_count=len(retained),
    )


def render_outcome_table(
    summary: OutcomeSummary,
    *,
    coverage_validated: bool,
    caption: str = (
        "Successful and failed experiment runs by dataset after excluding Qwen~3.5 27B. "
        "Each run is assigned to exactly one terminal outcome."
    ),
    label: str = EXPERIMENT_OUTCOMES_TABLE_LABEL,
) -> str:
    """Render a self-contained LaTeX table from an outcome summary."""
    if not summary.datasets:
        raise ValueError("At least one dataset is required to render the table")

    successful_counts = {
        dataset: sum(summary.count(outcome, dataset) for outcome in SUCCESS_OUTCOMES)
        for dataset in summary.datasets
    }
    failed_counts = {
        dataset: sum(summary.count(outcome, dataset) for outcome in FAILURE_OUTCOMES)
        for dataset in summary.datasets
    }
    dataset_totals = {
        dataset: successful_counts[dataset] + failed_counts[dataset]
        for dataset in summary.datasets
    }

    def cells(values: Mapping[str, int]) -> str:
        displayed = [_format_count(values[dataset]) for dataset in summary.datasets]
        displayed.append(_format_count(sum(values.values())))
        return " & ".join(displayed)

    def outcome_cells(outcome: str) -> str:
        values = {dataset: summary.count(outcome, dataset) for dataset in summary.datasets}
        return cells(values)

    headers = [DATASET_LABELS.get(dataset, dataset) for dataset in summary.datasets]
    note = "Complete coverage validated." if coverage_validated else "Coverage validation skipped."
    lines = [
        "% Generated by visualizations.experiment_outcomes; do not edit values manually.",
        (
            f"% Retained runs: {summary.retained_run_count}; "
            f"excluded-model runs: {summary.excluded_run_count}."
        ),
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{@{{}}l{'r' * (len(headers) + 1)}@{{}}}}",
        r"    \toprule",
        "    Outcome & " + " & ".join((*headers, "Total")) + LATEX_ROW_END,
        r"    \midrule",
        (
            f"    {OUTCOME_LABELS[SUCCESS_CLEAN]} & "
            f"{outcome_cells(SUCCESS_CLEAN)}" + LATEX_ROW_END
        ),
        (
            f"    {OUTCOME_LABELS[SUCCESS_AFTER_CORRECTION]} & "
            f"{outcome_cells(SUCCESS_AFTER_CORRECTION)}" + LATEX_ROW_END
        ),
        f"    \\textbf{{All successful runs}} & {cells(successful_counts)}" + LATEX_ROW_END,
        r"    \midrule",
    ]
    lines.extend(
        f"    {OUTCOME_LABELS[outcome]} & {outcome_cells(outcome)}" + LATEX_ROW_END
        for outcome in FAILURE_OUTCOMES
    )
    lines.extend(
        [
            f"    \\textbf{{All failed runs}} & {cells(failed_counts)}" + LATEX_ROW_END,
            r"    \midrule",
            f"    \\textbf{{Total runs}} & {cells(dataset_totals)}" + LATEX_ROW_END,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.4em}",
            f"  \\parbox{{\\linewidth}}{{\\scriptsize \\textit{{Note:}} {note}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_experiment_outcomes_table(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
    coverage_validated: bool,
) -> Path:
    """Calculate and write the terminal-outcomes table, returning its path."""
    summary = calculate_outcome_summary(
        results,
        excluded_model_markers=excluded_model_markers,
    )
    output_path = output_dir / EXPERIMENT_OUTCOMES_TABLE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_outcome_table(summary, coverage_validated=coverage_validated),
        encoding="utf-8",
    )
    return output_path


def model_is_excluded(model_id: object, markers: Sequence[str]) -> bool:
    """Return whether a model identifier contains one configured marker."""
    normalized = _text(model_id).casefold()
    return any(marker.casefold() in normalized for marker in markers)



def _method_for_record(record: pd.Series) -> str:
    pipeline = _text(record.get("pipeline"))
    fmt = _text(record.get("format"))
    if pipeline == "direct" and fmt in FORMATS:
        return fmt
    if pipeline == SQL_METHOD and not fmt:
        return SQL_METHOD
    raise ValueError(
        f"Unexpected pipeline/format combination: pipeline={pipeline!r}, format={fmt!r}"
    )


def _require_columns(results: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(results.columns)
    if missing:
        raise ValueError(f"Result data is missing required column(s): {', '.join(sorted(missing))}")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _is_true(value: object) -> bool:
    return value is True or (not _is_missing(value) and bool(value))


def _is_false(value: object) -> bool:
    return value is False or (not _is_missing(value) and not bool(value))


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _nonnegative_int(value: object) -> int:
    if _is_missing(value):
        return 0
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _format_count(value: int) -> str:
    return num(value)
