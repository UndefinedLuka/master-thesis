"""Result types produced by the two pipelines.

``PipelineResult`` is the unit of NDJSON output. The two pipelines fill different subsets:
Direct Context Inference leaves the Text-to-SQL fields ``None`` and vice versa. The scoring
pass (deferred) fills ``scores`` from ``raw_output`` / ``predicted_rows`` vs the reference.
"""

from __future__ import annotations

from typing import Any, TypedDict

from config.settings import PipelineKind
from llm.types import LLMMetrics
from metrics.types import AccuracyScores, ExecutionScore
from serialization.types import SerializationFormat

__all__ = ["PipelineKind", "RetryAttempt", "PipelineResult"]


class RetryAttempt(TypedDict):
    """One iteration of the Text-to-SQL error-correction loop."""

    attempt_index: int  # 0-based; 0 is the first try
    raw_text: str  # full model response this attempt (for post-hoc debugging)
    sql: str  # SQL extracted from the model output this attempt
    error: str | None  # execution error, or None if it ran cleanly
    llm_metrics: LLMMetrics  # generation cost of this attempt (retries cost tokens/time)


class PipelineResult(TypedDict):
    # identity
    example_id: str
    db_id: str
    dataset: str
    pipeline: PipelineKind
    format: SerializationFormat | None  # None for text_to_sql (schema-only)
    model_id: str

    # generation outputs
    raw_output: str  # the model's full final response text (both pipelines)
    predicted_sql: str | None  # SQL extracted from the final attempt; None for direct
    predicted_rows: list[dict[str, Any]] | None  # text_to_sql executed result; None for direct

    # operational efficiency (Chapter 3 §Operational efficiency)
    input_tokens: int
    output_tokens: int  # summed across retries for text_to_sql
    time_to_first_token_s: float
    reasoning_time_s: float  # <think>-phase time; 0.0 for non-thinking models; summed (t2sql)
    generation_time_s: float  # model time, summed across retries
    query_execution_time_s: float | None  # text_to_sql only
    total_time_s: float  # generation + query exec (Chapter 3 lines 196-199)

    # text_to_sql extras (Chapter 3 §Retrieval Accuracy, error-correction loop)
    retry_count: int  # number of retries beyond the first attempt (0..max_retries)
    attempts: list[RetryAttempt] | None
    execution_success: bool | None  # ran without error within the retry cap
    definitive_failure: bool | None  # still failing after max_retries

    # retrieval accuracy (filled by the scoring pass: score_results)
    scores: AccuracyScores | None  # normalized EM + ROUGE-L + BERTScore (both pipelines)
    execution: ExecutionScore | None  # binary execution success + result match (text_to_sql)

    # pipeline-level fatal error (generation failure, etc.)
    error: str | None
