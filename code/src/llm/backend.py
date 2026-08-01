"""The swappable LLM backend interface.

A :class:`Protocol` (structural typing) lets each backend stay a thin dataclass of methods
without inheriting a base class, matching the functional style. Swapping the inference
library later (e.g. adding vLLM) means one new module satisfying this Protocol plus one line
in :mod:`llm.factory` — nothing else changes.

Backend contract:
  * stream tokens and measure TTFT from submission to the first output token;
  * report split input/output token counts (never character counts — a v1 bug);
  * never raise to the caller: capture failures in ``LLMResponse.metrics["error"]``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm.types import BackendInfo, LLMRequest, LLMResponse


@runtime_checkable
class LLMBackend(Protocol):
    def info(self) -> BackendInfo:
        """Identifying metadata (backend name, model id, context length)."""
        ...

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one chat completion, returning normalized text + metrics."""
        ...

    def count_tokens(self, text: str) -> int:
        """Best available token count for ``text``.

        Tokenizer-accurate where the backend owns a tokenizer (llama_cpp); a conservative
        estimate otherwise (openai_compat). Used for context-fit checks and as a fallback
        when the server returns no usage block — treat as approximate, not exact.
        """
        ...

    def close(self) -> None:
        """Release the model / HTTP session. Must be idempotent."""
        ...
