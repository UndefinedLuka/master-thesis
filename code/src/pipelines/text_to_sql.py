"""Text-to-SQL pipeline with the error-correction loop.

Give the model the schema DDL, have it emit SQL, execute that SQL against the real database.
On execution error, feed the error back and regenerate — up to ``max_retries`` times
(Chapter 3 caps this at 3). We log how many retries each example needed; an example still
failing after the cap is a definitive failure.

Cost accounting (Chapter 3 §Operational efficiency): tokens and generation time are summed
across all attempts (retries cost resources too); query-execution time is summed across all
executions and added to total time, penalizing slow queries.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from datasets.types import DatasetExample
from llm.backend import LLMBackend
from llm.types import ChatMessage
from pipelines.shared import base_result, build_messages, make_request, model_id_of
from pipelines.types import PipelineResult, RetryAttempt
from prompts import render_prompt
from sqlite.connection import query_sqlite_safe

_SQL_FENCE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_KEYWORD = re.compile(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
# A keyword at the start of a line is a much stronger signal than one inside prose
# ("I would select the name..."), so the no-fence fallback prefers it.
_SQL_LINE_START = re.compile(
    r"^[ \t]*(SELECT|WITH|INSERT|UPDATE|DELETE)\b", re.IGNORECASE | re.MULTILINE
)


def extract_sql(raw_output: str) -> str:
    """Pull the SQL statement out of a model response.

    Prefers the first fenced code block containing a SQL keyword (a reasoning model may emit
    non-SQL fences first); falls back to keeping from the first SQL keyword onward. Returns a
    single statement, cut at the first semicolon that is outside any quoted string literal —
    a naive ``text.index(";")`` would truncate queries like ``WHERE note = 'a;b'``. Returns
    "" when the response contains no SQL at all (e.g. the sanctioned "Insufficient data in
    schema." refusal) so the caller can fail fast instead of executing prose against the
    database.
    """
    text = raw_output.strip()
    fences = [m.group(1).strip() for m in _SQL_FENCE.finditer(text)]
    sql_fences = [f for f in fences if _SQL_KEYWORD.search(f)]
    if sql_fences:
        text = sql_fences[0]
    match = _SQL_LINE_START.search(text) or _SQL_KEYWORD.search(text)
    if match is None:
        return ""  # no SQL anywhere (e.g. "Insufficient data in schema.")
    text = text[match.start() :].strip()
    end = _statement_end(text)
    if end is not None:
        text = text[: end + 1]
    return text.strip()


def _statement_end(sql: str) -> int | None:
    """Index of the first semicolon outside single/double-quoted literals, or None."""
    quote: str | None = None
    for index, char in enumerate(sql):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == ";":
            return index
    return None


def run_text_to_sql(
    backend: LLMBackend,
    example: DatasetExample,
    schema_ddl: str,
    db_path: Path,
    prompt_template: str,
    *,
    max_retries: int = 3,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> PipelineResult:
    """Run one Text-to-SQL example with the bounded error-correction loop."""
    result = base_result(example, pipeline="text_to_sql", fmt=None, model_id=model_id_of(backend))

    system_prompt = render_prompt(prompt_template, schema_ddl)
    messages: list[ChatMessage] = build_messages(
        system_prompt, example["question"], evidence=example["evidence"]
    )

    attempts: list[RetryAttempt] = []
    total_generation_s = 0.0
    total_reasoning_s = 0.0
    total_query_s = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    ttft = 0.0
    final_raw = ""
    final_sql = ""
    predicted_rows: list[dict[str, Any]] | None = None
    execution_success = False
    fatal_error: str | None = None

    # 1 initial attempt + up to max_retries corrections.
    for attempt_index in range(max_retries + 1):
        response = backend.generate(
            make_request(messages, max_tokens=max_tokens, temperature=temperature, seed=seed)
        )
        metrics = response.metrics

        total_generation_s += metrics["total_generation_time_s"]
        total_reasoning_s += metrics["reasoning_time_s"]
        total_input_tokens += metrics["input_tokens"]
        total_output_tokens += metrics["output_tokens"]
        if attempt_index == 0:
            ttft = metrics["time_to_first_token_s"]

        if metrics["error"] is not None:
            # Generation itself failed (not a SQL error) — abort the loop.
            fatal_error = metrics["error"]
            attempts.append(
                RetryAttempt(
                    attempt_index=attempt_index,
                    raw_text=response.text,
                    sql="",
                    error=fatal_error,
                    llm_metrics=metrics,
                )
            )
            break

        sql = extract_sql(response.text)
        final_raw = response.text
        final_sql = sql

        if not sql:
            # No SQL in the reply (refusal / prose). Don't execute prose against the DB;
            # count the attempt and feed the problem back like any other error.
            exec_error: str | None = "no SQL statement found in the response"
            qr = None
        else:
            query_start = time.perf_counter()
            qr, exec_error = query_sqlite_safe(db_path, sql)
            total_query_s += time.perf_counter() - query_start

        attempts.append(
            RetryAttempt(
                attempt_index=attempt_index,
                raw_text=response.text,
                sql=sql,
                error=exec_error,
                llm_metrics=metrics,
            )
        )

        if exec_error is None and qr is not None:
            predicted_rows = qr.rows
            execution_success = True
            break

        # Feed the error back for the next correction attempt.
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": _correction_prompt(exec_error or "unknown error")},
        ]

    retry_count = max(0, len(attempts) - 1)

    result["raw_output"] = final_raw
    result["predicted_sql"] = final_sql
    result["predicted_rows"] = predicted_rows
    result["input_tokens"] = total_input_tokens
    result["output_tokens"] = total_output_tokens
    result["time_to_first_token_s"] = ttft
    result["reasoning_time_s"] = total_reasoning_s
    result["generation_time_s"] = total_generation_s
    result["query_execution_time_s"] = total_query_s
    result["total_time_s"] = total_generation_s + total_query_s
    result["retry_count"] = retry_count
    result["attempts"] = attempts
    result["execution_success"] = execution_success
    result["definitive_failure"] = (not execution_success) and (fatal_error is None)
    result["error"] = fatal_error
    return result


def _correction_prompt(error: str) -> str:
    return (
        "The SQL query failed to execute against the database with the following error:\n"
        f"{error}\n\n"
        "Fix the query. Return only the corrected SQL."
    )
