"""Tests for the stacked average result-time visualization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.average_time_by_format import (  # noqa: E402
    ATTEMPT_REASONING_COLUMN,
    ATTEMPT_TTFT_COLUMN,
    calculate_average_times,
    plot_average_times,
    save_figure,
)
from visualizations.data import load_scored_results  # noqa: E402


def test_loader_projects_metrics_from_every_retry_attempt(tmp_path: Path) -> None:
    path = tmp_path / "sql.scored.ndjson"
    record = {
        "pipeline": "text_to_sql",
        "format": None,
        "attempts": [
            {
                "llm_metrics": {
                    "time_to_first_token_s": 0.25,
                    "reasoning_time_s": 0.5,
                }
            },
            {
                "llm_metrics": {
                    "time_to_first_token_s": 0.75,
                    "reasoning_time_s": 1.5,
                }
            },
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = load_scored_results(
        path,
        columns=(ATTEMPT_TTFT_COLUMN, ATTEMPT_REASONING_COLUMN),
    )

    assert loaded.loc[0, ATTEMPT_TTFT_COLUMN] == [0.25, 0.75]
    assert loaded.loc[0, ATTEMPT_REASONING_COLUMN] == [0.5, 1.5]


def test_calculate_average_times_uses_non_overlapping_phases_and_sql_retries() -> None:
    results = pd.DataFrame(
        {
            "pipeline": ["direct", "direct", "direct", "text_to_sql"],
            "format": ["csv", "csv", "xml", None],
            "time_to_first_token_s": [2.0, 4.0, 1.0, 0.5],
            "reasoning_time_s": [3.0, 1.0, 0.0, 1.0],
            "generation_time_s": [10.0, 9.0, 4.0, 12.0],
            "query_execution_time_s": [None, None, None, 2.0],
            ATTEMPT_TTFT_COLUMN: [None, None, None, [0.5, 1.5]],
            ATTEMPT_REASONING_COLUMN: [None, None, None, [1.0, 2.0]],
        }
    )

    summary = calculate_average_times(results)
    indexed = summary.set_index("method")

    assert indexed.loc["csv", "ingestion_time_s"] == pytest.approx(3.0)
    assert indexed.loc["csv", "thinking_time_s"] == pytest.approx(2.0)
    assert indexed.loc["csv", "generation_time_s"] == pytest.approx(4.5)
    assert indexed.loc["csv", "sql_execution_time_s"] == 0.0
    assert indexed.loc["csv", "total_time_s"] == pytest.approx(9.5)

    # SQL uses both retries' phase times: ingestion=0.5+1.5, thinking=1+2,
    # generation residual=12-2-3, then adds two seconds of query execution.
    assert indexed.loc["SQL execution", "ingestion_time_s"] == pytest.approx(2.0)
    assert indexed.loc["SQL execution", "thinking_time_s"] == pytest.approx(3.0)
    assert indexed.loc["SQL execution", "generation_time_s"] == pytest.approx(7.0)
    assert indexed.loc["SQL execution", "sql_execution_time_s"] == pytest.approx(2.0)
    assert indexed.loc["SQL execution", "total_time_s"] == pytest.approx(14.0)
    assert summary["method"].tolist() == ["csv", "xml", "SQL execution"]


def test_calculate_average_times_rejects_overlapping_invalid_source_times() -> None:
    results = pd.DataFrame(
        {
            "pipeline": ["direct"],
            "format": ["json"],
            "time_to_first_token_s": [4.0],
            "reasoning_time_s": [3.0],
            "generation_time_s": [5.0],
            "query_execution_time_s": [None],
        }
    )

    with pytest.raises(ValueError, match="timing phases longer"):
        calculate_average_times(results)


def test_stacked_time_plot_can_be_saved(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "method": ["csv", "xml", "SQL execution"],
            "ingestion_time_s": [3.0, 5.0, 1.0],
            "thinking_time_s": [1.0, 2.0, 0.5],
            "generation_time_s": [4.0, 6.0, 2.0],
            "sql_execution_time_s": [0.0, 0.0, 0.25],
            "answer_count": [2, 2, 2],
            "total_time_s": [8.0, 13.0, 3.75],
        }
    )

    figure, axes = plot_average_times(summary)
    output = tmp_path / "nested" / "average-times.png"
    save_figure(figure, output)

    assert axes.get_title() == "Average Time to Result by Method"
    assert axes.get_xlabel() == "Method"
    assert [label.get_text() for label in axes.get_xticklabels()][-2:] == [
        "XML",
        "SQL",
    ]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Ingestion",
        "Thinking",
        "Generation",
        "SQL execution",
    ]
    assert len(axes.containers) == 4
    labels = [text.get_text() for text in axes.texts]
    assert {"3.00s", "1.00s", "4.00s", "Total 8.00s"}.issubset(labels)
    assert any(
        "Ingestion   1.00s" in label
        and "Thinking    0.50s" in label
        and "Generation  2.00s" in label
        and "SQL query   0.250s" in label
        for label in labels
    )
    assert output.is_file()
    assert output.stat().st_size > 0
