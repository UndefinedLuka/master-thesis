"""Tests for the Text-to-SQL retry-effectiveness visualization."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.retry_effectiveness import (  # noqa: E402
    RetryEffectiveness,
    calculate_retry_effectiveness,
    plot_retry_effectiveness,
    save_figure,
)


def test_calculate_retry_effectiveness_tracks_recovery_and_correctness() -> None:
    results = pd.DataFrame(
        {
            "pipeline": ["text_to_sql"] * 8 + ["direct"],
            "retry_count": [0, 0, 1, 1, 2, 3, 3, 3, 0],
            "execution_success": [True, True, True, True, True, True, False, False, True],
            "execution.result_match": [True, False, True, False, True, False, False, False, True],
        }
    )

    analysis = calculate_retry_effectiveness(results)
    attempts = analysis.attempts.set_index("attempt_number")

    assert analysis.total_runs == 8
    assert analysis.final_failure_count == 2
    assert attempts.loc[1, "reached_count"] == 8
    assert attempts.loc[1, "execution_success_count"] == 2
    assert attempts.loc[1, "correct_count"] == 1
    assert attempts.loc[1, "incorrect_count"] == 1
    assert attempts.loc[1, "execution_recovery_rate"] == pytest.approx(2 / 8)
    assert attempts.loc[1, "correct_among_success_rate"] == pytest.approx(1 / 2)

    assert attempts.loc[2, "reached_count"] == 6
    assert attempts.loc[2, "execution_success_count"] == 2
    assert attempts.loc[2, "correct_count"] == 1
    assert attempts.loc[3, "reached_count"] == 4
    assert attempts.loc[3, "execution_success_count"] == 1
    assert attempts.loc[3, "correct_count"] == 1
    assert attempts.loc[4, "reached_count"] == 3
    assert attempts.loc[4, "execution_success_count"] == 1
    assert attempts.loc[4, "correct_count"] == 0
    assert attempts.loc[4, "correct_among_success_rate"] == 0.0


def test_calculate_retry_effectiveness_rejects_missing_success_diagnostics() -> None:
    results = pd.DataFrame(
        {
            "pipeline": ["text_to_sql"],
            "retry_count": [0],
            "execution_success": [True],
            "execution.result_match": [None],
        }
    )

    with pytest.raises(ValueError, match="result_match"):
        calculate_retry_effectiveness(results)


def test_retry_effectiveness_plot_has_count_and_rate_panels(tmp_path: Path) -> None:
    attempts = pd.DataFrame(
        {
            "attempt_number": [1, 2, 3, 4],
            "attempt_label": ["Initial\nattempt", "Retry 1", "Retry 2", "Retry 3"],
            "reached_count": [100, 30, 10, 4],
            "execution_success_count": [70, 15, 4, 1],
            "correct_count": [50, 10, 3, 0],
            "incorrect_count": [20, 5, 1, 1],
            "execution_recovery_rate": [0.70, 0.50, 0.40, 0.25],
            "correct_among_success_rate": [50 / 70, 10 / 15, 0.75, 0.0],
        }
    )
    analysis = RetryEffectiveness(attempts, total_runs=110, final_failure_count=20)

    figure, axes = plot_retry_effectiveness(analysis, datasets=("bird", "spider"))
    output = tmp_path / "nested" / "retry-effectiveness.png"
    save_figure(figure, output)

    count_axis, rate_axis = axes
    assert count_axis.get_title() == "Outcomes by Resolving Attempt"
    assert count_axis.get_yscale() == "log"
    assert rate_axis.get_title() == "Effectiveness at Each Attempt"
    assert rate_axis.get_ylabel() == "Rate (%)"
    assert rate_axis.get_ylim()[1] == pytest.approx(112)
    assert [label.get_text() for label in count_axis.get_xticklabels()][-1] == "Never\nexecuted"
    figure_text = " ".join(text.get_text() for text in figure.texts)
    assert "Retries recovered 20/30 runs (66.7%)" in figure_text
    assert "13 recovered results (65.0%) were correct" in figure_text
    assert output.is_file()
    assert output.stat().st_size > 0
