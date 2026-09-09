"""Tests for the Chapter 5 terminal experiment-outcome table generator."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")

import pandas as pd

from serialization.types import FORMATS
from visualizations.experiment_outcomes import (
    FAILED_CONTEXT,
    FAILED_DIRECT_OUTPUT,
    FAILED_GENERATION,
    FAILED_NO_SQL,
    FAILED_OTHER,
    FAILED_SQL,
    OUTCOME_COLUMNS,
    SUCCESS_AFTER_CORRECTION,
    SUCCESS_CLEAN,
    OutcomeSummary,
    calculate_outcome_summary,
    classify_outcome,
    generate_experiment_outcomes_table,
    render_outcome_table,
    validate_complete_coverage,
)


def _result(
    *,
    dataset: str = "spider",
    pipeline: str = "direct",
    fmt: str | None = "json",
    model: str = "model.gguf",
    example_id: str = "spider:db:0",
    predicted_sql: str | None = None,
    retry_count: int = 0,
    execution_success: bool | None = None,
    definitive_failure: bool | None = None,
    error: str | None = None,
    has_answer: bool = True,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "format": fmt,
        "model_id": model,
        "example_id": example_id,
        "predicted_sql": predicted_sql,
        "retry_count": retry_count,
        "execution_success": execution_success,
        "definitive_failure": definitive_failure,
        "error": error,
        "has_answer": has_answer,
    }


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_result(), SUCCESS_CLEAN),
        (_result(has_answer=False), FAILED_DIRECT_OUTPUT),
        (_result(error="context_exceeded", has_answer=False), FAILED_CONTEXT),
        (_result(error="CUDA failure", has_answer=False), FAILED_GENERATION),
        (
            _result(
                pipeline="text_to_sql",
                fmt=None,
                predicted_sql="SELECT 1",
                execution_success=True,
            ),
            SUCCESS_CLEAN,
        ),
        (
            _result(
                pipeline="text_to_sql",
                fmt=None,
                predicted_sql="SELECT 1",
                retry_count=2,
                execution_success=True,
            ),
            SUCCESS_AFTER_CORRECTION,
        ),
        (
            _result(
                pipeline="text_to_sql",
                fmt=None,
                predicted_sql="SELECT * FROM missing",
                retry_count=3,
                execution_success=False,
                definitive_failure=True,
            ),
            FAILED_SQL,
        ),
        (
            _result(
                pipeline="text_to_sql",
                fmt=None,
                predicted_sql="",
                retry_count=3,
                execution_success=False,
                definitive_failure=True,
            ),
            FAILED_NO_SQL,
        ),
        (
            _result(
                pipeline="text_to_sql",
                fmt=None,
                execution_success=None,
            ),
            FAILED_OTHER,
        ),
    ],
)
def test_classify_outcome_assigns_each_terminal_state(
    record: dict[str, object], expected: str
) -> None:
    assert classify_outcome(record) == expected


def test_summary_excludes_qwen_27b_and_aggregates_by_dataset() -> None:
    results = pd.DataFrame(
        [
            _result(dataset="spider"),
            _result(dataset="bird", example_id="bird:db:0", error="context_exceeded"),
            _result(
                dataset="bird",
                example_id="bird:db:1",
                pipeline="text_to_sql",
                fmt=None,
                predicted_sql="SELECT 1",
                retry_count=1,
                execution_success=True,
            ),
            _result(model="Qwen3.5-27B-Q4_K_M.gguf"),
        ],
        columns=OUTCOME_COLUMNS,
    )

    summary = calculate_outcome_summary(results)

    assert summary.retained_run_count == 3
    assert summary.excluded_run_count == 1
    assert summary.retained_model_count == 1
    assert summary.count(SUCCESS_CLEAN, "spider") == 1
    assert summary.count(FAILED_CONTEXT, "bird") == 1
    assert summary.count(SUCCESS_AFTER_CORRECTION, "bird") == 1


def _complete_matrix() -> pd.DataFrame:
    rows = [
        _result(fmt=fmt, example_id="spider:db:0")
        for fmt in FORMATS
    ]
    rows.append(
        _result(
            pipeline="text_to_sql",
            fmt=None,
            example_id="spider:db:0",
            predicted_sql="SELECT 1",
            execution_success=True,
        )
    )
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def test_complete_coverage_accepts_all_six_formats_and_sql() -> None:
    coverage = validate_complete_coverage(
        _complete_matrix(),
        datasets=("spider",),
        excluded_model_markers=(),
        expected_question_counts={"spider": 1},
        expected_model_count=1,
    )

    assert coverage.question_counts == {"spider": 1}
    assert coverage.model_count == 1
    assert coverage.run_count == 7


def test_complete_coverage_rejects_a_missing_method() -> None:
    incomplete = _complete_matrix().iloc[:-1].copy()

    with pytest.raises(ValueError, match="Incomplete method coverage"):
        validate_complete_coverage(
            incomplete,
            datasets=("spider",),
            excluded_model_markers=(),
            expected_question_counts={"spider": 1},
            expected_model_count=1,
        )



def test_render_outcome_table_includes_totals_and_writes_file(tmp_path: Path) -> None:
    summary = OutcomeSummary(
        counts={
            SUCCESS_CLEAN: {"spider": 3, "bird": 4},
            SUCCESS_AFTER_CORRECTION: {"spider": 1, "bird": 2},
            FAILED_CONTEXT: {"spider": 5, "bird": 6},
            FAILED_SQL: {"spider": 0, "bird": 1},
            FAILED_NO_SQL: {"spider": 0, "bird": 0},
            FAILED_GENERATION: {"spider": 0, "bird": 0},
            FAILED_DIRECT_OUTPUT: {"spider": 0, "bird": 0},
            FAILED_OTHER: {"spider": 0, "bird": 0},
        },
        datasets=("spider", "bird"),
        retained_run_count=22,
        excluded_run_count=7,
        retained_model_count=1,
    )

    latex = render_outcome_table(summary, coverage_validated=True)
    output = generate_experiment_outcomes_table(
        pd.DataFrame(
            [
                _result(dataset="spider"),
                _result(dataset="bird", example_id="bird:db:0"),
            ],
            columns=OUTCOME_COLUMNS,
        ),
        tmp_path / "tables",
        excluded_model_markers=(),
        coverage_validated=False,
    )

    assert r"Successful without a recorded error & 3 & 4 & 7 \\" in latex
    assert r"\textbf{All successful runs} & 4 & 6 & 10 \\" in latex
    assert r"\textbf{All failed runs} & 5 & 7 & 12 \\" in latex
    assert r"\textbf{Total runs} & 9 & 13 & 22 \\" in latex
    assert "Complete coverage validated." in latex
    assert "Coverage validation skipped." in output.read_text(encoding="utf-8")
