"""The benchmark record — one NDJSON row per (question, run-cell).

This is the single type that legitimately depends on every layer, so it lives at
the top level rather than in any one package. ``to_json_dict`` is explicit
because ``dataclasses.asdict`` would not convert ``Enum`` members to their
``.value`` nor ``Path`` to ``str``; a hand-written serializer keeps the on-disk
schema stable and analysis-friendly.

Each record carries reproducibility metadata (host OS, parameter scale,
quantization, seed, model-load time) so results from different machines — where
TTFT/throughput are not comparable (Metal vs CUDA vs CPU) — are never silently
pooled during analysis.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

from .config import RunConfig
from .datasets.types import QuestionItem
from .enums import DatasetKind, PipelineKind, SerializationFormat
from .eval.types import EvalScores
from .llm.base import LLMUsage
from .pipelines.types import PipelineResult, SqlAttempt


@dataclass(frozen=True, slots=True)
class HostInfo:
    """Where a record was produced. TTFT/throughput are only comparable within
    the same host, so this is recorded per record.
    """

    os: str
    machine: str
    python_version: str

    @classmethod
    def detect(cls) -> HostInfo:
        return cls(
            os=platform.system(),
            machine=platform.machine(),
            python_version=platform.python_version(),
        )


def _usage_to_dict(usage: LLMUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "ttft_s": usage.ttft_s,
        "total_generation_s": usage.total_generation_s,
        "load_duration_s": usage.load_duration_s,
        "prompt_eval_s": usage.prompt_eval_s,
        "eval_s": usage.eval_s,
        "tokens_per_second": usage.tokens_per_second,
    }


def _attempt_to_dict(attempt: SqlAttempt) -> dict[str, Any]:
    return {
        "attempt_index": attempt.attempt_index,
        "sql": attempt.sql,
        "outcome": attempt.outcome.value,
        "db_error": attempt.db_error,
        "exec_time_s": attempt.exec_time_s,
        "llm_usage": _usage_to_dict(attempt.llm_usage),
    }


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """A single evaluated question for one matrix cell."""

    # Identity / provenance
    question_id: str
    db_id: str
    dataset: DatasetKind
    fmt: SerializationFormat
    pipeline: PipelineKind
    model_id: str
    param_scale_b: float
    quantization: str
    seed: int
    sample_rows: int | None
    host: HostInfo

    # Inputs
    question: str
    gold_sql: str
    reference_text: str

    # Outputs
    result: PipelineResult
    scores: EvalScores

    @classmethod
    def build(
        cls,
        *,
        run: RunConfig,
        item: QuestionItem,
        reference_text: str,
        result: PipelineResult,
        scores: EvalScores,
        host: HostInfo,
    ) -> BenchmarkRecord:
        return cls(
            question_id=item.question_id,
            db_id=item.db_id,
            dataset=run.dataset.kind,
            fmt=run.fmt,
            pipeline=run.pipeline,
            model_id=run.model.model_id,
            param_scale_b=run.model.param_scale_b,
            quantization=run.model.quantization,
            seed=run.model.options.seed,
            sample_rows=run.dataset.sample_rows,
            host=host,
            question=item.question,
            gold_sql=item.gold_sql,
            reference_text=reference_text,
            result=result,
            scores=scores,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe dict. Enums -> ``.value``; nested dataclasses expanded
        by hand so the schema is explicit and stable.
        """
        sql_outcome = self.result.sql_outcome
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "dataset": self.dataset.value,
            "format": self.fmt.value,
            "format_family": self.fmt.family.value,
            "pipeline": self.pipeline.value,
            "model_id": self.model_id,
            "param_scale_b": self.param_scale_b,
            "quantization": self.quantization,
            "seed": self.seed,
            "sample_rows": self.sample_rows,
            "host": {
                "os": self.host.os,
                "machine": self.host.machine,
                "python_version": self.host.python_version,
            },
            "question": self.question,
            "gold_sql": self.gold_sql,
            "reference_text": self.reference_text,
            "result": {
                "answer_text": self.result.answer_text,
                "raw_model_output": self.result.raw_model_output,
                "total_usage": _usage_to_dict(self.result.total_usage),
                "total_latency_s": self.result.total_latency_s,
                "execution_ok": self.result.execution_ok,
                "error": self.result.error,
                "sql_outcome": (
                    None
                    if sql_outcome is None
                    else {
                        "status": sql_outcome.status.value,
                        "retry_count": sql_outcome.retry_count,
                        "final_sql": sql_outcome.final_sql,
                        "attempts": [
                            _attempt_to_dict(a) for a in sql_outcome.attempts
                        ],
                    }
                ),
            },
            "scores": {
                "rouge_l": self.scores.rouge_l,
                "bertscore_f1": self.scores.bertscore_f1,
                "execution_accuracy": self.scores.execution_accuracy,
                "row_set_match": self.scores.row_set_match,
            },
        }
