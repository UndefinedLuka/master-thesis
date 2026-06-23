"""Pipeline behavior using the fake LLM client and the in-memory sample."""

from __future__ import annotations

import sqlite3

import pytest

from thesis_bench.config import GenerationOptions
from thesis_bench.datasets.types import QuestionItem
from thesis_bench.db import sqlite
from thesis_bench.enums import (
    SerializationFormat,
    SqlAttemptOutcome,
    TextToSqlStatus,
)
from thesis_bench.pipelines.base import PipelineContext
from thesis_bench.pipelines.direct import DirectPipeline
from thesis_bench.pipelines.text_to_sql import TextToSqlPipeline, extract_sql
from thesis_bench.serialization.snapshot import DatabaseSnapshot, Row

from .fakes import FakeLLMClient


class _ConnExecutor:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str) -> tuple[tuple[str, ...], tuple[Row, ...]]:
        return sqlite.execute_on_connection(self._conn, sql)


def _ctx(
    snapshot: DatabaseSnapshot,
    client: FakeLLMClient,
    conn: sqlite3.Connection,
    max_retries: int = 3,
) -> PipelineContext:
    return PipelineContext(
        item=QuestionItem(
            question_id="q1", db_id="demo", question="How many?", gold_sql="SELECT 1"
        ),
        snapshot=snapshot,
        fmt=SerializationFormat.JSON,
        client=client,
        options=GenerationOptions(),
        executor=_ConnExecutor(conn),
        max_sql_retries=max_retries,
    )


def test_direct_pipeline_returns_model_text(people_snapshot: DatabaseSnapshot) -> None:
    client = FakeLLMClient(["Alice is the oldest."])
    with sqlite.materialize_sample(people_snapshot) as conn:
        result = DirectPipeline().execute(_ctx(people_snapshot, client, conn))
    assert result.answer_text == "Alice is the oldest."
    assert result.execution_ok is None
    assert result.sql_outcome is None
    assert result.total_usage.output_tokens > 0


def test_text_to_sql_success_first_try(people_snapshot: DatabaseSnapshot) -> None:
    client = FakeLLMClient(["```sql\nSELECT name FROM people ORDER BY age DESC\n```"])
    with sqlite.materialize_sample(people_snapshot) as conn:
        result = TextToSqlPipeline().execute(_ctx(people_snapshot, client, conn))
    assert result.execution_ok is True
    assert result.sql_outcome is not None
    assert result.sql_outcome.status is TextToSqlStatus.SUCCESS
    assert result.sql_outcome.retry_count == 0
    assert "Carol" in result.answer_text  # oldest


def test_text_to_sql_retries_then_succeeds(people_snapshot: DatabaseSnapshot) -> None:
    client = FakeLLMClient(
        [
            "```sql\nSELECT name FROM nonexistent\n```",  # fails
            "```sql\nSELECT name FROM people\n```",  # succeeds
        ]
    )
    with sqlite.materialize_sample(people_snapshot) as conn:
        result = TextToSqlPipeline().execute(_ctx(people_snapshot, client, conn))
    assert result.execution_ok is True
    assert result.sql_outcome is not None
    assert result.sql_outcome.retry_count == 1
    assert len(result.sql_outcome.attempts) == 2
    assert result.sql_outcome.attempts[0].outcome is SqlAttemptOutcome.EXECUTION_ERROR
    # Usage accumulates across both attempts.
    assert result.total_usage.output_tokens > 0


def test_text_to_sql_definitive_failure(people_snapshot: DatabaseSnapshot) -> None:
    client = FakeLLMClient(["```sql\nSELECT * FROM nope\n```"])  # always fails
    with sqlite.materialize_sample(people_snapshot) as conn:
        result = TextToSqlPipeline().execute(
            _ctx(people_snapshot, client, conn, max_retries=2)
        )
    assert result.execution_ok is False
    assert result.sql_outcome is not None
    assert result.sql_outcome.status is TextToSqlStatus.FAILED_AFTER_RETRIES
    # 1 initial + 2 retries = 3 attempts.
    assert len(result.sql_outcome.attempts) == 3
    assert result.sql_outcome.retry_count == 2


def test_text_to_sql_generation_failure(people_snapshot: DatabaseSnapshot) -> None:
    client = FakeLLMClient(["Insufficient data in schema."])  # no SQL
    with sqlite.materialize_sample(people_snapshot) as conn:
        result = TextToSqlPipeline().execute(_ctx(people_snapshot, client, conn))
    assert result.execution_ok is False
    assert result.sql_outcome is not None
    assert result.sql_outcome.attempts[0].outcome is SqlAttemptOutcome.GENERATION_ERROR


@pytest.mark.parametrize(
    "text,expected",
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("SELECT a FROM t", "SELECT a FROM t"),
        ("no sql here", None),
        ("Insufficient data in schema.", None),
    ],
)
def test_extract_sql(text: str, expected: str | None) -> None:
    assert extract_sql(text) == expected
