"""Text-to-SQL retry loop, using a mock client (no network/GPU)."""

from __future__ import annotations

from pathlib import Path

from benchmark.datasets import Question
from benchmark.llm_client import GenResult
from benchmark.pipelines.text_to_sql import extract_sql, run_sql_question
from benchmark.reference import Reference
from benchmark.serialize import get_db_schema
from benchmark.sqlite_io import query_sqlite, to_records


class MockClient:
    """Returns a scripted sequence of GenResults, one per generate() call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def generate(self, system: str, user: str) -> GenResult:
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return GenResult(
            text=text,
            input_tokens=10,
            output_tokens=5,
            ttft_s=0.01,
            total_generation_s=0.05,
            finish_reason="stop",
        )


def _question() -> Question:
    return Question(index=0, db_id="tiny", question="How many pets does Alice own?", gold_sql="")


def _reference(db: Path) -> Reference:
    gold = "SELECT COUNT(*) FROM pets WHERE owner = 1"
    rows = to_records(query_sqlite(db, gold))
    from benchmark.metrics import rows_to_text

    return Reference(0, "tiny", "q", gold, rows, rows_to_text(rows), None)


def test_extract_sql_from_fence() -> None:
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2; SELECT 3") == "SELECT 2"


def test_success_first_try(tiny_db: Path) -> None:
    schema = get_db_schema(tiny_db)
    client = MockClient(["```sql\nSELECT COUNT(*) FROM pets WHERE owner = 1\n```"])
    row = run_sql_question(client, "m", "spider", _question(), schema, tiny_db, _reference(tiny_db))
    assert row.execution_accuracy is True
    assert row.definitive_failure is False
    assert row.retry_count == 0
    assert row.result_set_match is True
    assert client.calls == 1


def test_recovers_after_error(tiny_db: Path) -> None:
    schema = get_db_schema(tiny_db)
    client = MockClient(
        [
            "```sql\nSELECT * FROM nonexistent_table\n```",  # fails
            "```sql\nSELECT COUNT(*) FROM pets WHERE owner = 1\n```",  # fixes
        ]
    )
    row = run_sql_question(client, "m", "spider", _question(), schema, tiny_db, _reference(tiny_db))
    assert row.execution_accuracy is True
    assert row.definitive_failure is False
    assert row.retry_count == 1
    assert row.input_tokens == 20  # summed across both attempts


def test_definitive_failure_after_cap(tiny_db: Path) -> None:
    schema = get_db_schema(tiny_db)
    client = MockClient(["```sql\nSELECT * FROM nope\n```"])  # always fails
    row = run_sql_question(
        client, "m", "spider", _question(), schema, tiny_db, _reference(tiny_db), retry_cap=3
    )
    assert row.execution_accuracy is False
    assert row.definitive_failure is True
    assert row.retry_count == 3  # initial + 3 retries = 4 attempts, last index 3
    assert client.calls == 4


def test_valid_but_wrong_query(tiny_db: Path) -> None:
    schema = get_db_schema(tiny_db)
    # Runs fine, but returns the wrong count (owner 2, not 1).
    client = MockClient(["```sql\nSELECT COUNT(*) FROM pets WHERE owner = 2\n```"])
    row = run_sql_question(client, "m", "spider", _question(), schema, tiny_db, _reference(tiny_db))
    assert row.execution_accuracy is True
    assert row.result_set_match is False
