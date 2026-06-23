"""Text-to-SQL pipeline with the error-correction loop.

Flow (ch-03):
1. Prompt the model with the schema only; it generates a SQL query.
2. Extract the SQL from the response and execute it against the materialized
   sample (the same rows the direct pipeline saw).
3. On a database error, feed schema + failing query + error message back to the
   model and retry — up to ``max_sql_retries`` times. After that, it is a
   definitive failure.

Per-attempt cost (tokens, generation time, execution time) is recorded on each
``SqlAttempt`` and accumulated into the run-level totals. Query execution time is
folded into ``total_latency_s`` (only for this pipeline), which penalizes models
that emit slow queries.
"""

from __future__ import annotations

import re
import time

from ..enums import PipelineKind, SqlAttemptOutcome, TextToSqlStatus
from ..errors import SqlExecutionError
from ..eval.accuracy import rows_to_text
from ..llm.base import ChatMessage, LLMUsage
from ..prompts.loader import PromptTemplate, build_prompt
from ..serialization.snapshot import Row
from .base import Pipeline, PipelineContext
from .types import PipelineResult, SqlAttempt, TextToSqlOutcome

# Matches a fenced ```sql ... ``` block (preferred) or any fenced block.
_SQL_FENCE_RE = re.compile(
    r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL
)


def extract_sql(text: str) -> str | None:
    """Pull a SQL statement out of the model's response.

    Prefers a fenced code block; otherwise falls back to the first text that
    looks like a statement (starts with a SQL keyword). Returns ``None`` if no
    plausible SQL is present — the caller treats that as a generation error.
    """
    match = _SQL_FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.strip().rstrip(";").strip()
    if not candidate:
        return None
    if not re.match(r"(?is)^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b", candidate):
        return None
    return candidate


class TextToSqlPipeline(Pipeline):
    kind = PipelineKind.TEXT_TO_SQL

    def execute(self, ctx: PipelineContext) -> PipelineResult:
        schema = ctx.snapshot.schema_sql
        attempts: list[SqlAttempt] = []
        total_usage = LLMUsage.zero()
        total_latency = 0.0

        # Attempt 0 uses the generation prompt; retries use the correction prompt.
        last_sql: str | None = None
        last_error: str | None = None
        final_rows: tuple[Row, ...] | None = None
        final_columns: tuple[str, ...] = ()
        status = TextToSqlStatus.FAILED_AFTER_RETRIES

        max_attempts = ctx.max_sql_retries + 1
        for attempt_index in range(max_attempts):
            if attempt_index == 0:
                system_prompt = build_prompt(PromptTemplate.SQL_ANSWER, schema=schema)
                user_content = ctx.item.question
            else:
                system_prompt = build_prompt(
                    PromptTemplate.SQL_RETRY,
                    schema=schema,
                    previous_sql=last_sql or "",
                    error=last_error or "",
                )
                user_content = ctx.item.question

            response = ctx.client.chat(
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_content),
                ],
                options=ctx.options,
            )
            total_usage = total_usage + response.usage
            total_latency += response.usage.total_generation_s

            if response.error is not None:
                attempts.append(
                    SqlAttempt(
                        attempt_index=attempt_index,
                        sql=None,
                        outcome=SqlAttemptOutcome.GENERATION_ERROR,
                        db_error=response.error,
                        llm_usage=response.usage,
                        exec_time_s=None,
                    )
                )
                last_error = response.error
                last_sql = None
                status = TextToSqlStatus.FAILED_GENERATION
                # A backend failure won't fix itself by retrying with an error
                # prompt; stop here.
                break

            sql = extract_sql(response.text)
            if sql is None:
                attempts.append(
                    SqlAttempt(
                        attempt_index=attempt_index,
                        sql=None,
                        outcome=SqlAttemptOutcome.GENERATION_ERROR,
                        db_error="No extractable SQL in model response.",
                        llm_usage=response.usage,
                        exec_time_s=None,
                    )
                )
                last_error = "No extractable SQL in model response."
                last_sql = None
                status = TextToSqlStatus.FAILED_GENERATION
                continue

            last_sql = sql
            exec_start = time.perf_counter()
            try:
                columns, rows = ctx.executor.execute(sql)
            except SqlExecutionError as exc:
                exec_time = time.perf_counter() - exec_start
                total_latency += exec_time
                attempts.append(
                    SqlAttempt(
                        attempt_index=attempt_index,
                        sql=sql,
                        outcome=SqlAttemptOutcome.EXECUTION_ERROR,
                        db_error=exc.db_message,
                        llm_usage=response.usage,
                        exec_time_s=exec_time,
                    )
                )
                last_error = exc.db_message
                status = TextToSqlStatus.FAILED_AFTER_RETRIES
                continue

            # Success.
            exec_time = time.perf_counter() - exec_start
            total_latency += exec_time
            attempts.append(
                SqlAttempt(
                    attempt_index=attempt_index,
                    sql=sql,
                    outcome=SqlAttemptOutcome.EXECUTED_OK,
                    db_error=None,
                    llm_usage=response.usage,
                    exec_time_s=exec_time,
                )
            )
            final_rows = rows
            final_columns = columns
            status = TextToSqlStatus.SUCCESS
            break

        outcome = TextToSqlOutcome(
            status=status,
            attempts=tuple(attempts),
            final_sql=last_sql,
            final_rows=final_rows,
        )

        execution_ok = status is TextToSqlStatus.SUCCESS
        answer_text = (
            rows_to_text(final_columns, final_rows)
            if execution_ok and final_rows is not None
            else ""
        )

        return PipelineResult(
            answer_text=answer_text,
            raw_model_output=last_sql or "",
            total_usage=total_usage,
            total_latency_s=total_latency,
            execution_ok=execution_ok,
            sql_outcome=outcome,
            error=None if execution_ok else last_error,
        )
