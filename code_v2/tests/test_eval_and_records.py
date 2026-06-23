"""Eval helpers, canonical stringification, and record serialization."""

from __future__ import annotations

import json
from pathlib import Path

from thesis_bench.config import (
    DatasetConfig,
    GenerationOptions,
    ModelConfig,
    RunConfig,
)
from thesis_bench.datasets.types import GroundTruth, QuestionItem
from thesis_bench.enums import (
    Backend,
    DatasetKind,
    PipelineKind,
    SerializationFormat,
    SqlAttemptOutcome,
    TextToSqlStatus,
)
from thesis_bench.eval.accuracy import row_set_match, rows_to_text
from thesis_bench.eval.metrics import rouge_l, score_record
from thesis_bench.llm.base import LLMUsage
from thesis_bench.pipelines.types import PipelineResult, SqlAttempt, TextToSqlOutcome
from thesis_bench.records import BenchmarkRecord, HostInfo


def test_rows_to_text_is_order_independent() -> None:
    cols = ("a", "b")
    rows_a = ({"a": 1, "b": 2}, {"a": 3, "b": 4})
    rows_b = ({"a": 3, "b": 4}, {"a": 1, "b": 2})
    assert rows_to_text(cols, rows_a) == rows_to_text(cols, rows_b)


def test_row_set_match_ignores_order_and_alias() -> None:
    assert row_set_match(
        ("x",), ({"x": 1}, {"x": 2}), ("y",), ({"y": 2}, {"y": 1})
    )
    assert not row_set_match(("x",), ({"x": 1},), ("y",), ({"y": 2},))


def test_rouge_l_bounds() -> None:
    assert rouge_l("same text", "same text") == 1.0
    assert rouge_l("", "") == 1.0
    assert rouge_l("a", "") == 0.0


def _run() -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            backend=Backend.OLLAMA,
            model_id="m:7b-q8_0",
            param_scale_b=7.0,
            quantization="Q8_0",
            options=GenerationOptions(seed=5),
        ),
        dataset=DatasetConfig(
            kind=DatasetKind.SPIDER,
            questions_path=Path("q.json"),
            db_root=Path("db"),
        ),
        fmt=SerializationFormat.JSON,
        pipeline=PipelineKind.TEXT_TO_SQL,
    )


def test_record_serializes_to_json() -> None:
    outcome = TextToSqlOutcome(
        status=TextToSqlStatus.SUCCESS,
        attempts=(
            SqlAttempt(
                attempt_index=0,
                sql="SELECT 1",
                outcome=SqlAttemptOutcome.EXECUTED_OK,
                db_error=None,
                llm_usage=LLMUsage.zero(),
                exec_time_s=0.001,
            ),
        ),
        final_sql="SELECT 1",
        final_rows=({"1": 1},),
    )
    result = PipelineResult(
        answer_text="1",
        raw_model_output="SELECT 1",
        total_usage=LLMUsage(10, 2, 0.01, 0.05, 0.0, 0.02, 0.03),
        total_latency_s=0.05,
        execution_ok=True,
        sql_outcome=outcome,
    )
    gt = GroundTruth(rows=({"1": 1},), reference_text="1")
    scores = score_record(result, gt, PipelineKind.TEXT_TO_SQL)

    record = BenchmarkRecord.build(
        run=_run(),
        item=QuestionItem("spider-0", "db", "q?", "SELECT 1"),
        reference_text="1",
        result=result,
        scores=scores,
        host=HostInfo.detect(),
    )
    payload = record.to_json_dict()
    # Must be JSON-serializable (enums -> values, no dataclass leakage).
    text = json.dumps(payload)
    reloaded = json.loads(text)
    assert reloaded["format"] == "json"
    assert reloaded["pipeline"] == "text_to_sql"
    assert reloaded["seed"] == 5
    assert reloaded["result"]["sql_outcome"]["retry_count"] == 0
    assert reloaded["scores"]["execution_accuracy"] is True
    assert reloaded["scores"]["row_set_match"] is True
    assert reloaded["scores"]["rouge_l"] == 1.0


def test_score_record_direct_has_no_sql_fields() -> None:
    result = PipelineResult(
        answer_text="hello world",
        raw_model_output="hello world",
        total_usage=LLMUsage.zero(),
        total_latency_s=0.0,
    )
    gt = GroundTruth(rows=(), reference_text="hello world")
    scores = score_record(result, gt, PipelineKind.DIRECT)
    assert scores.execution_accuracy is None
    assert scores.row_set_match is None
    assert scores.rouge_l == 1.0
