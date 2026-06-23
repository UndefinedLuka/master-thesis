"""The swappable LLM backend boundary.

``LLMClient`` is a ``Protocol`` (structural typing) so a future ``LlamaCppClient``
or ``LlamaServerClient`` satisfies it without importing or subclassing anything
here, and so tests can supply a trivial fake. The pipelines depend only on this
interface — never on Ollama directly — which is what makes the backend choice
reversible (the user's explicit requirement).

The metric fields on ``LLMUsage`` are deliberately rich: v1 measured tokens by
character count and let lazy model-load time pollute TTFT. Here we record the
backend's *native* token counters and keep ``load_duration`` separate so those
bugs cannot recur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..config import GenerationOptions

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat-turn message. ``role`` is a ``Literal`` because it is an external
    wire contract (the chat API's), not a domain concept we branch on.
    """

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Resource accounting for a single completion.

    ``input_tokens``/``output_tokens`` come from the backend's native counters
    (Ollama: ``prompt_eval_count`` / ``eval_count``), never from string length.

    Durations are in seconds. ``ttft_s`` is measured client-side at the first
    content-bearing chunk. ``load_duration_s`` is tracked separately so the
    one-time model load never inflates TTFT or generation time.
    """

    input_tokens: int
    output_tokens: int
    ttft_s: float
    total_generation_s: float  # prompt-eval + eval time (model time, excl. load)
    load_duration_s: float
    prompt_eval_s: float
    eval_s: float

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.eval_s if self.eval_s > 0 else 0.0

    @classmethod
    def zero(cls) -> LLMUsage:
        return cls(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __add__(self, other: LLMUsage) -> LLMUsage:
        """Accumulate usage across attempts (Text-to-SQL retry loop).

        Token counts and durations sum. TTFT is taken from the *first* attempt
        (the earliest non-zero value), since "time to first token" is only
        meaningful for the initial response.
        """
        ttft = self.ttft_s if self.ttft_s > 0 else other.ttft_s
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            ttft_s=ttft,
            total_generation_s=self.total_generation_s + other.total_generation_s,
            load_duration_s=self.load_duration_s + other.load_duration_s,
            prompt_eval_s=self.prompt_eval_s + other.prompt_eval_s,
            eval_s=self.eval_s + other.eval_s,
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A completed (non-streaming-to-the-caller) chat completion."""

    text: str
    usage: LLMUsage
    stop_reason: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class LLMClient(Protocol):
    """Minimal backend surface the pipelines depend on."""

    @property
    def model_id(self) -> str: ...

    def chat(
        self, messages: list[ChatMessage], options: GenerationOptions
    ) -> LLMResponse:
        """Run a chat completion and return the full text plus usage metrics.

        Implementations should capture TTFT client-side and read native token
        counters; on backend failure they return an ``LLMResponse`` with
        ``error`` set rather than raising, so the orchestrator records the failure
        and continues.
        """
        ...
