"""Tests for SQL extraction and the Text-to-SQL error-correction loop.

A scripted fake backend (satisfying the LLMBackend protocol structurally) lets us drive the
retry loop deterministically without a real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from datasets.types import DatasetExample
from llm.backend import LLMBackend
from llm.types import BackendInfo, LLMRequest, LLMResponse, empty_metrics
from pipelines.direct import run_direct_inference
from pipelines.text_to_sql import extract_sql, run_text_to_sql


@dataclass(slots=True)
class FakeBackend:
    """Returns a pre-scripted list of responses, one per generate() call."""

    responses: list[str]
    _calls: int = field(default=0)

    def info(self) -> BackendInfo:
        return BackendInfo(name="fake", model_id="fake-model")

    def generate(self, request: LLMRequest) -> LLMResponse:
        text = self.responses[min(self._calls, len(self.responses) - 1)]
        self._calls += 1
        metrics = empty_metrics()
        metrics["input_tokens"] = 10
        metrics["output_tokens"] = 5
        metrics["time_to_first_token_s"] = 0.01
        metrics["total_generation_time_s"] = 0.1
        return LLMResponse(text=text, metrics=metrics)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def close(self) -> None:
        pass


def _example() -> DatasetExample:
    return DatasetExample(
        dataset="spider",
        db_id="tiny",
        question="How many singers?",
        gold_sql="SELECT count(*) FROM singer",
        evidence=None,
        difficulty=None,
        example_id="spider:tiny:0",
    )


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(FakeBackend(responses=["x"]), LLMBackend)


def test_extract_sql_from_fenced_block() -> None:
    raw = "Here is the query:\n```sql\nSELECT * FROM singer;\n```\nDone."
    assert extract_sql(raw) == "SELECT * FROM singer;"


def test_extract_sql_without_fence() -> None:
    raw = "SELECT name FROM singer WHERE id = 1"
    assert extract_sql(raw).startswith("SELECT name FROM singer")


def test_extract_sql_skips_non_sql_fence() -> None:
    raw = (
        "Let me think:\n```text\nsingers are in the singer table\n```\n"
        "```sql\nSELECT count(*) FROM singer;\n```"
    )
    assert extract_sql(raw) == "SELECT count(*) FROM singer;"


def test_extract_sql_returns_empty_for_prose_refusal() -> None:
    assert extract_sql("Insufficient data in schema.") == ""
    assert extract_sql("I cannot answer this question.") == ""


def test_extract_sql_keeps_semicolon_inside_string_literal() -> None:
    raw = "```sql\nSELECT * FROM t WHERE note = 'a;b';\n```"
    assert extract_sql(raw) == "SELECT * FROM t WHERE note = 'a;b';"


def test_extract_sql_truncates_at_first_statement() -> None:
    raw = "```sql\nSELECT 1; SELECT 2;\n```"
    assert extract_sql(raw) == "SELECT 1;"


def test_extract_sql_handles_with_cte() -> None:
    raw = "WITH c AS (SELECT id FROM singer) SELECT count(*) FROM c"
    assert extract_sql(raw).startswith("WITH c AS")


def test_extract_sql_prefers_line_start_keyword_over_prose() -> None:
    # "select" inside prose must not win over the actual statement on its own line.
    raw = "First I will select the right table.\nSELECT name FROM singer"
    assert extract_sql(raw) == "SELECT name FROM singer"


def test_text_to_sql_refusal_is_not_executed(tiny_db: Path) -> None:
    backend = FakeBackend(responses=["Insufficient data in schema."])
    result = run_text_to_sql(backend, _example(), "ddl", tiny_db, "<data>", max_retries=1)
    assert result["execution_success"] is False
    assert result["definitive_failure"] is True
    assert result["predicted_rows"] is None
    assert all("no SQL statement" in a["error"] for a in result["attempts"])


def test_direct_pipeline_records_metrics(tiny_db: Path) -> None:
    backend = FakeBackend(responses=["There are 5 singers."])
    serialized = '{"singer": [{"id": 1, "name": "Alice"}]}'
    result = run_direct_inference(backend, _example(), serialized, "json", "schema: <data>")
    assert result["pipeline"] == "direct"
    assert result["format"] == "json"
    assert result["raw_output"] == "There are 5 singers."
    assert result["input_tokens"] == 10
    assert result["total_time_s"] == result["generation_time_s"]
    assert result["query_execution_time_s"] is None


def test_text_to_sql_success_first_try(tiny_db: Path) -> None:
    backend = FakeBackend(responses=["```sql\nSELECT count(*) AS c FROM singer\n```"])
    result = run_text_to_sql(backend, _example(), "CREATE TABLE singer(...)", tiny_db, "<data>")
    assert result["execution_success"] is True
    assert result["retry_count"] == 0
    assert result["definitive_failure"] is False
    assert result["predicted_rows"] == [{"c": 5}]
    assert result["query_execution_time_s"] is not None
    assert result["total_time_s"] >= result["generation_time_s"]


def test_text_to_sql_recovers_after_one_retry(tiny_db: Path) -> None:
    backend = FakeBackend(
        responses=[
            "SELECT * FROM wrong_table",  # fails
            "SELECT count(*) AS c FROM singer",  # corrected
        ]
    )
    result = run_text_to_sql(backend, _example(), "ddl", tiny_db, "<data>", max_retries=3)
    assert result["execution_success"] is True
    assert result["retry_count"] == 1
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["error"] is not None
    assert result["attempts"][1]["error"] is None
    # tokens accumulate across attempts (10 in / 5 out per call, 2 calls)
    assert result["input_tokens"] == 20
    assert result["output_tokens"] == 10


def test_text_to_sql_definitive_failure(tiny_db: Path) -> None:
    backend = FakeBackend(responses=["SELECT * FROM still_wrong"])
    result = run_text_to_sql(backend, _example(), "ddl", tiny_db, "<data>", max_retries=3)
    assert result["execution_success"] is False
    assert result["definitive_failure"] is True
    assert result["retry_count"] == 3  # initial + 3 retries = 4 attempts
    assert len(result["attempts"]) == 4
    assert result["predicted_rows"] is None


def test_text_to_sql_records_raw_output_and_extracted_sql(tiny_db: Path) -> None:
    raw = "Here you go:\n```sql\nSELECT count(*) AS c FROM singer\n```"
    backend = FakeBackend(responses=[raw])
    result = run_text_to_sql(backend, _example(), "ddl", tiny_db, "<data>")
    assert result["raw_output"] == raw  # the full model response, for post-hoc debugging
    assert result["predicted_sql"] == "SELECT count(*) AS c FROM singer"
    assert result["attempts"][0]["raw_text"] == raw


def test_text_to_sql_correction_message_carries_error(tiny_db: Path) -> None:
    captured_requests: list[LLMRequest] = []

    @dataclass(slots=True)
    class RecordingBackend(FakeBackend):
        def generate(self, request: LLMRequest) -> LLMResponse:
            captured_requests.append(request)
            return FakeBackend.generate(self, request)

    backend = RecordingBackend(
        responses=["SELECT * FROM wrong_table", "SELECT count(*) AS c FROM singer"]
    )
    result = run_text_to_sql(backend, _example(), "ddl", tiny_db, "<data>", max_retries=1)
    assert result["execution_success"] is True
    # The retry's message list must contain the actual sqlite error text.
    retry_messages = captured_requests[1].messages
    assert any("wrong_table" in m["content"] for m in retry_messages)


def test_text_to_sql_fatal_generation_error_aborts_loop(tiny_db: Path) -> None:
    @dataclass(slots=True)
    class ErroringBackend(FakeBackend):
        def generate(self, request: LLMRequest) -> LLMResponse:
            self._calls += 1
            metrics = empty_metrics()
            metrics["error"] = "server exploded"
            return LLMResponse(text="", metrics=metrics)

    backend = ErroringBackend(responses=[])
    result = run_text_to_sql(backend, _example(), "ddl", tiny_db, "<data>", max_retries=3)
    assert backend._calls == 1  # no pointless retries against a dead backend
    assert result["error"] == "server exploded"
    assert result["execution_success"] is False
    # Not a definitive SQL failure — the model never got to answer.
    assert result["definitive_failure"] is False
