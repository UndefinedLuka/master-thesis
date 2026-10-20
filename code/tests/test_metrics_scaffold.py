"""Tests for dependency-free binary execution accuracy."""

from __future__ import annotations

from metrics.execution import execution_accuracy


def test_successful_execution_scores_one() -> None:
    score = execution_accuracy([{"c": 5}])
    assert score == {"execution_success": True, "execution_accuracy": 1.0}


def test_successful_empty_result_scores_one() -> None:
    score = execution_accuracy([])
    assert score == {"execution_success": True, "execution_accuracy": 1.0}


def test_failed_execution_scores_zero() -> None:
    score = execution_accuracy(None)
    assert score == {"execution_success": False, "execution_accuracy": 0.0}
