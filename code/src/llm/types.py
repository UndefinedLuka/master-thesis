"""Backend-agnostic request / response / metric types for the LLM wrapper.

These normalized types are the contract between pipelines and any concrete backend, so no
vendor SDK type ever leaks upward. ``LLMMetrics`` carries exactly the operational-efficiency
numbers Chapter 3 requires (split input/output tokens, TTFT, total generation time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A normalized chat-completion request.

    Streaming is not a request option: every backend always streams, because TTFT can only
    be measured from a token stream (part of the backend contract in :mod:`llm.backend`).
    """

    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 0.0  # deterministic by default
    seed: int | None = None
    stop: list[str] | None = None


class LLMMetrics(TypedDict):
    """Per-call operational-efficiency metrics (serializes cleanly to NDJSON)."""

    input_tokens: int
    output_tokens: int
    time_to_first_token_s: float  # submission -> first output token
    reasoning_time_s: float  # first token -> end of <think> block; 0.0 for non-thinking models
    total_generation_time_s: float  # model time only (query exec added later for Text-to-SQL)
    tokens_per_second: float
    stop_reason: str | None  # "stop" | "length" | backend-specific
    error: str | None  # None on success; str(exception) otherwise


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized result: the full assistant text plus its metrics."""

    text: str
    metrics: LLMMetrics


def empty_metrics() -> LLMMetrics:
    """A zeroed metrics record, used as the starting point / on early failure."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "time_to_first_token_s": 0.0,
        "reasoning_time_s": 0.0,
        "total_generation_time_s": 0.0,
        "tokens_per_second": 0.0,
        "stop_reason": None,
        "error": None,
    }


def derive_tokens_per_second(metrics: LLMMetrics) -> float:
    """Generation throughput: output tokens over (total time - TTFT)."""
    generation_time = metrics["total_generation_time_s"] - metrics["time_to_first_token_s"]
    if generation_time > 0 and metrics["output_tokens"] > 0:
        return metrics["output_tokens"] / generation_time
    return 0.0


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Identifying metadata about a constructed backend."""

    name: str  # "llama_cpp" | "openai_compat"
    model_id: str
