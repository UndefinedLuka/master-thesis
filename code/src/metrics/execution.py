"""Binary Execution Accuracy scoring and result diagnostics for Text-to-SQL.

Execution Accuracy follows the thesis definition: 1 when the generated query executes
successfully within the retry cap, otherwise 0. Result equality against the gold rows is
reported separately as ``result_match``. Both ``rows_to_canonical`` and
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
    """Return binary query-execution accuracy plus a separate result-match diagnostic.

    ``predicted_rows`` is ``None`` when the query never executed successfully within the retry
    cap. Execution Accuracy is therefore 1 when ``predicted_rows`` is present (including an
    empty result set) and 0 otherwise; whether those rows are correct does not affect it.

    ``result_match`` compares predicted rows with usable gold rows. It is false when either
    side is unavailable. Row order is ignored by default and preserved when
    ``order_sensitive=True``.
    """
    execution_success = predicted_rows is not None
    result_match = False
    if predicted_rows is not None and gold_rows is not None:
        predicted = rows_to_canonical(predicted_rows, order_sensitive=order_sensitive)
        gold = rows_to_canonical(gold_rows, order_sensitive=order_sensitive)
        result_match = predicted == gold

    return ExecutionScore(
        execution_success=execution_success,
        result_match=result_match,
        execution_accuracy=1.0 if execution_success else 0.0,
    )
