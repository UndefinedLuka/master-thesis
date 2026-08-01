"""Tests for the dependency-free metrics: rows_to_canonical and execution_accuracy."""

from __future__ import annotations

from metrics.execution import execution_accuracy, rows_to_canonical


def test_rows_to_canonical_set_equality_ignores_order() -> None:
    a = rows_to_canonical([{"x": 1}, {"x": 2}])
    b = rows_to_canonical([{"x": 2}, {"x": 1}])
    assert a == b


def test_rows_to_canonical_stringifies_cells() -> None:
    # 1 (int) and "1" (str) should canonicalize equal — matches BIRD's dirty-data spirit
    assert rows_to_canonical([{"x": 1}]) == rows_to_canonical([{"x": "1"}])


def test_rows_to_canonical_numeric_representations_equal() -> None:
    # SQL engines disagree on int-vs-float rendering; 1, 1.0 and "1.0" must compare equal
    assert rows_to_canonical([{"x": 1}]) == rows_to_canonical([{"x": 1.0}])
    assert rows_to_canonical([{"x": "1.0"}]) == rows_to_canonical([{"x": 1}])
    assert rows_to_canonical([{"x": 0.5}]) == rows_to_canonical([{"x": ".5"}])
    # but genuinely different numbers and non-numeric strings stay distinct
    assert rows_to_canonical([{"x": 1}]) != rows_to_canonical([{"x": 2}])
    assert rows_to_canonical([{"x": "abc"}]) != rows_to_canonical([{"x": "abd"}])


def test_rows_to_canonical_order_sensitive() -> None:
    a = rows_to_canonical([{"x": 1}, {"x": 2}], order_sensitive=True)
    b = rows_to_canonical([{"x": 2}, {"x": 1}], order_sensitive=True)
    assert a != b


def test_rows_to_canonical_none() -> None:
    assert rows_to_canonical(None) is None


def test_execution_accuracy_match() -> None:
    score = execution_accuracy([{"c": 5}], [{"count(*)": 5}])
    # column names differ but the value tuple matches (Spider-style EX compares values)
    assert score["execution_success"] is True
    assert score["result_match"] is True
    assert score["execution_accuracy"] == 1.0


def test_execution_accuracy_mismatch() -> None:
    score = execution_accuracy([{"c": 4}], [{"c": 5}])
    assert score["execution_success"] is True
    assert score["result_match"] is False
    assert score["execution_accuracy"] == 0.0


def test_execution_accuracy_ignores_row_order_by_default() -> None:
    score = execution_accuracy([{"x": 2}, {"x": 1}], [{"x": 1}, {"x": 2}])
    assert score["execution_accuracy"] == 1.0


def test_execution_accuracy_order_sensitive() -> None:
    score = execution_accuracy([{"x": 2}, {"x": 1}], [{"x": 1}, {"x": 2}], order_sensitive=True)
    assert score["result_match"] is False


def test_execution_accuracy_failed_query_scores_zero() -> None:
    # predicted_rows is None => the query never executed (definitive failure)
    score = execution_accuracy(None, [{"x": 1}])
    assert score["execution_success"] is False
    assert score["execution_accuracy"] == 0.0


def test_execution_accuracy_gold_error_scores_zero() -> None:
    # gold rows None => reference gold SQL itself errored; no verdict possible
    score = execution_accuracy([{"x": 1}], None)
    assert score["execution_accuracy"] == 0.0
