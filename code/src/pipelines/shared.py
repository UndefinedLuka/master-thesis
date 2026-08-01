"""Helpers shared by both pipeline runners."""

from __future__ import annotations

from datasets.types import DatasetExample
from llm.backend import LLMBackend
from llm.types import ChatMessage, LLMRequest
from pipelines.types import PipelineKind, PipelineResult
from serialization.types import SerializationFormat


def build_messages(
    system_prompt: str, question: str, *, evidence: str | None = None
) -> list[ChatMessage]:
    """Assemble the chat messages: system prompt, then the question.

    BIRD's external-knowledge ``evidence`` (when present) is appended to the question, since
    Chapter 3 notes BIRD requires external knowledge for correct retrieval.
    """
    user_content = question if not evidence else f"{question}\n\nHint: {evidence}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def make_request(
    messages: list[ChatMessage],
    *,
    max_tokens: int | None,
    temperature: float,
    seed: int | None,
) -> LLMRequest:
    """Build the normalized request both pipelines send to the backend."""
    return LLMRequest(messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed)


def base_result(
    example: DatasetExample,
    *,
    pipeline: PipelineKind,
    fmt: SerializationFormat | None,
    model_id: str,
) -> PipelineResult:
    """A zeroed :class:`PipelineResult` with identity fields filled and the rest defaulted."""
    return PipelineResult(
        example_id=example["example_id"],
        db_id=example["db_id"],
        dataset=example["dataset"],
        pipeline=pipeline,
        format=fmt,
        model_id=model_id,
        raw_output="",
        predicted_sql=None,
        predicted_rows=None,
        input_tokens=0,
        output_tokens=0,
        time_to_first_token_s=0.0,
        reasoning_time_s=0.0,
        generation_time_s=0.0,
        query_execution_time_s=None,
        total_time_s=0.0,
        retry_count=0,
        attempts=None,
        execution_success=None,
        definitive_failure=None,
        scores=None,
        execution=None,
        error=None,
    )


def model_id_of(backend: LLMBackend) -> str:
    """The backend's model identifier, recorded on every result for traceability."""
    return backend.info().model_id
