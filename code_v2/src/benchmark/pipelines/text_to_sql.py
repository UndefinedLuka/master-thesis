"""Text-to-SQL pipeline with the 3-retry error-correction loop.

The model is given the SQL schema and asked for a query. The query is executed
against the real SQLite database; on a database error the error message is fed
back to the model for up to ``retry_cap`` retries. Per the thesis:

* token counts and generation time are **summed across all attempts** (each
  retry is real cost);
* the successful query's **execution time is added** to the total generation
  time (a query is not "answered" until its rows are retrieved);
* ``retry_count`` is logged; failure after the cap is a ``definitive_failure``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from benchmark.datasets import Question
from benchmark.llm_client import GenResult, LLMClient
from benchmark.metrics import compute_rouge_l, result_set_match, rows_to_text
from benchmark.prompts import build_sql_prompt
from benchmark.reference import Reference
from benchmark.results import ResultRow
from benchmark.sqlite_io import query_sqlite, to_records

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL statement out of a model response.

    Prefers the first fenced ```sql block; otherwise strips prose and takes the
    text up to the first semicolon (or the whole thing if there is none).
    """
    m = _SQL_FENCE.search(text)
    candidate = m.group(1) if m else text
    candidate = candidate.strip()
    # Take the first statement only.
    if ";" in candidate:
        candidate = candidate.split(";", 1)[0]
    return candidate.strip()


@dataclass
class SqlAttempt:
    attempt_no: int
    sql: str
    gen: GenResult
    exec_error: str | None
    exec_time_s: float


def run_sql_question(
    client: LLMClient,
    model_name: str,
    dataset: str,
    q: Question,
    schema: str,
    db_path,
    ref: Reference | None,
    retry_cap: int = 3,
) -> ResultRow:
    system = build_sql_prompt(schema, q.evidence)

    attempts: list[SqlAttempt] = []
    error_feedback: str | None = None
    success_rows: list[dict] | None = None
    final_sql = ""
    final_exec_time = 0.0

    for attempt_no in range(retry_cap + 1):  # 1 initial + up to retry_cap retries
        if error_feedback is None:
            user = q.question
        else:
            user = (
                f"{q.question}\n\n"
                f"The previous SQL query failed with this database error:\n{error_feedback}\n\n"
                f"Return a corrected single SQL query."
            )

        gen = client.generate(system, user)
        if gen.error is not None:
            # API/transport failure: record and stop retrying (not a SQL fault).
            attempts.append(SqlAttempt(attempt_no, "", gen, gen.error, 0.0))
            break

        sql = extract_sql(gen.text)
        t0 = time.perf_counter()
        try:
            df = query_sqlite(db_path, sql)
            exec_time = time.perf_counter() - t0
            success_rows = to_records(df)
            final_sql = sql
            final_exec_time = exec_time
            attempts.append(SqlAttempt(attempt_no, sql, gen, None, exec_time))
            break
        except Exception as e:  # noqa: BLE001 - SQL error drives the retry loop
            exec_time = time.perf_counter() - t0
            error_feedback = str(e)
            final_sql = sql
            attempts.append(SqlAttempt(attempt_no, sql, gen, error_feedback, exec_time))

    success = success_rows is not None
    retry_count = attempts[-1].attempt_no if attempts else 0

    # Operational totals summed across every attempt (+ successful exec time).
    input_tokens = sum(a.gen.input_tokens for a in attempts)
    output_tokens = sum(a.gen.output_tokens for a in attempts)
    gen_time = sum(a.gen.total_generation_s for a in attempts)
    ttft = attempts[0].gen.ttft_s if attempts else 0.0
    total_generation_s = gen_time + final_exec_time

    reference_unavailable = ref is None or ref.error is not None
    output_text = rows_to_text(success_rows) if success else ""
    reference_text = ref.answer_text if ref else ""

    rouge = 0.0
    rs_match = False
    if success and not reference_unavailable:
        rouge = compute_rouge_l(output_text, reference_text)
        rs_match = result_set_match(success_rows, ref.rows if ref else None)

    last_error = None if success else (attempts[-1].exec_error if attempts else "no attempts")

    return ResultRow(
        model=model_name,
        dataset=dataset,
        pipeline="sql",
        format=None,
        question_index=q.index,
        db_id=q.db_id,
        question=q.question,
        output_text=output_text,
        reference_text=reference_text,
        reference_unavailable=reference_unavailable,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_s=ttft,
        total_generation_s=total_generation_s,
        rouge_l=rouge,
        final_sql=final_sql,
        retry_count=retry_count,
        definitive_failure=not success,
        execution_accuracy=success,
        result_set_match=rs_match,
        exec_time_s=final_exec_time,
        error=last_error,
        attempts=[
            {
                "attempt_no": a.attempt_no,
                "sql": a.sql,
                "input_tokens": a.gen.input_tokens,
                "output_tokens": a.gen.output_tokens,
                "ttft_s": a.gen.ttft_s,
                "generation_s": a.gen.total_generation_s,
                "exec_time_s": a.exec_time_s,
                "exec_error": a.exec_error,
                "api_error": a.gen.error,
            }
            for a in attempts
        ],
    )
