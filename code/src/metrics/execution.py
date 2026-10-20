"""Binary Execution Accuracy scoring for Text-to-SQL."""

from __future__ import annotations

from typing import Any

from metrics.types import ExecutionScore


def execution_accuracy(predicted_rows: list[dict[str, Any]] | None) -> ExecutionScore:
    """Return binary execution success (a successful empty result counts as success).

    ``predicted_rows`` is ``None`` when the query did not execute successfully within the
    retry cap. Gold rows are deliberately not involved in this execution metric.
    """
    execution_success = predicted_rows is not None
    return ExecutionScore(
        execution_success=execution_success,
        execution_accuracy=1.0 if execution_success else 0.0,
    )
