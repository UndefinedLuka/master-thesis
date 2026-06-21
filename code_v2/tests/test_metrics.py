"""Metric helpers: rows_to_text and order-insensitive result-set matching."""

from __future__ import annotations

from benchmark.metrics import compute_rouge_l, result_set_match, rows_to_text


def test_rows_to_text_is_order_insensitive() -> None:
    a = [{"x": 1}, {"x": 2}]
    b = [{"x": 2}, {"x": 1}]
    assert rows_to_text(a) == rows_to_text(b)


def test_rows_to_text_empty() -> None:
    assert rows_to_text(None) == ""
    assert rows_to_text([]) == ""


def test_result_set_match_ignores_column_labels_and_order() -> None:
    pred = [{"c": 2}, {"c": 1}]
    gold = [{"count": 1}, {"count": 2}]
    assert result_set_match(pred, gold) is True
    assert result_set_match(pred, [{"count": 1}]) is False
    assert result_set_match(None, gold) is False


def test_rouge_l_identical_and_disjoint() -> None:
    assert compute_rouge_l("the cat sat", "the cat sat") == 1.0
    assert compute_rouge_l("", "") == 1.0
    assert compute_rouge_l("abc", "") == 0.0
