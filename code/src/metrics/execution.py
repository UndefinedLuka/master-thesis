"""Execution Accuracy scoring for Text-to-SQL (Spider-style EX).

The Text-to-SQL *pipeline* already records ``execution_success`` and ``predicted_rows``; this
module adds the comparison-against-gold verdict. Both ``rows_to_canonical`` and
``execution_accuracy`` are pure and dependency-free.
"""

from __future__ import annotations

import math
from typing import Any

from metrics.types import ExecutionScore


def _canonical_cell(value: Any) -> str:
    """Normalize one cell to a comparable string.

    Numeric-looking values are canonicalized through ``float`` (rounded to 1e-6) so ``1``,
    ``"1"`` and ``1.0`` all compare equal — SQL engines and gold queries frequently disagree
    on int vs float rendering, a classic EX-metric pitfall. Everything else compares as its
    string form (matching the dirty-data spirit of BIRD).
    """
    if value is None:
        return ""
    if isinstance(value, bool):  # bool is an int subclass; keep True != 1 out of scope
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return str(value)
    return repr(round(number, 6))


def rows_to_canonical(
    rows: list[dict[str, Any]] | None, *, order_sensitive: bool = False
) -> list[tuple[str, ...]] | None:
    """Normalize result rows for comparison.

    Each row becomes a tuple of canonicalized cell values (numeric values compare
    representation-independently — see ``_canonical_cell``). Row order is preserved when
    ``order_sensitive`` else sorted, so set-equality ignores ordering by default.
    """
    if rows is None:
        return None
    canonical = [tuple(_canonical_cell(value) for value in row.values()) for row in rows]
    return canonical if order_sensitive else sorted(canonical)


def execution_accuracy(
    predicted_rows: list[dict[str, Any]] | None,
    gold_rows: list[dict[str, Any]] | None,
    *,
    order_sensitive: bool = False,
) -> ExecutionScore:
    """Compare executed rows to the gold result set (Spider-style EX).

    ``predicted_rows`` is ``None`` when the query never executed (definitive failure); that is
    ``execution_success=False`` and scores 0. The gold side being ``None`` means the reference
    gold SQL itself errored, so no verdict is possible and the example also scores 0.

    Default policy is set-equality of stringified rows (column order within a row is preserved,
    row order is not). Pass ``order_sensitive=True`` only when the gold SQL has ``ORDER BY``.
    """
    execution_success = predicted_rows is not None
    if predicted_rows is None or gold_rows is None:
        return ExecutionScore(
            execution_success=execution_success,
            result_match=False,
            execution_accuracy=0.0,
        )
    predicted = rows_to_canonical(predicted_rows, order_sensitive=order_sensitive)
    gold = rows_to_canonical(gold_rows, order_sensitive=order_sensitive)
    result_match = predicted == gold
    return ExecutionScore(
        execution_success=execution_success,
        result_match=result_match,
        execution_accuracy=1.0 if result_match else 0.0,
    )
