"""Pipeline-layer value types, including the Text-to-SQL retry-loop model.

The retry loop (ch-03: "up to three compilation retries… logs the exact number
of retries") is modelled as an immutable sequence of ``SqlAttempt`` plus a
terminal ``TextToSqlOutcome``. Modelling each attempt explicitly means the exact
retry count, per-attempt errors, and per-attempt cost are all auditable rather
than collapsed into a single number.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..enums import SqlAttemptOutcome, TextToSqlStatus
from ..llm.base import LLMUsage
from ..serialization.snapshot import Row


@dataclass(frozen=True, slots=True)
class SqlAttempt:
    """One generate-and-execute attempt within the error-correction loop.

    ``attempt_index`` is 0-based: attempt 0 is the first generation, attempts
    1..N are retries triggered by feeding the previous ``db_error`` back to the
    model. ``llm_usage`` and ``exec_time_s`` are *this attempt's* costs; the
    pipeline accumulates them into the run-level totals.
    """

    attempt_index: int
    sql: str | None  # None when extraction/generation itself failed
    outcome: SqlAttemptOutcome
    db_error: str | None  # sqlite error fed back on the next retry
    llm_usage: LLMUsage
    exec_time_s: float | None  # query execution time, if it ran


@dataclass(frozen=True, slots=True)
class TextToSqlOutcome:
    """Terminal result of the full attempt sequence.

    ``retry_count == len(attempts) - 1`` by construction: a first-try success is
    ``0`` retries; the maximum sequence is 1 initial attempt + N retries.
    """

    status: TextToSqlStatus
    attempts: tuple[SqlAttempt, ...]
    final_sql: str | None
    final_rows: tuple[Row, ...] | None

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """The common shape both pipelines produce, so the orchestrator and eval path
    are identical regardless of pipeline.

    ``answer_text`` is what gets scored against the ground truth: for the direct
    pipeline it is the model's raw natural-language answer; for Text-to-SQL it is
    the canonical stringification of the executed result rows.

    ``total_usage`` and ``total_latency_s`` are accumulated across all attempts
    (a single attempt for the direct pipeline). Per ch-03, Text-to-SQL folds
    query execution time into ``total_latency_s``.

    The ``execution_*`` / ``sql_outcome`` fields are ``None`` for the direct
    pipeline.
    """

    answer_text: str
    raw_model_output: str
    total_usage: LLMUsage
    total_latency_s: float
    execution_ok: bool | None = None
    sql_outcome: TextToSqlOutcome | None = None
    error: str | None = None
