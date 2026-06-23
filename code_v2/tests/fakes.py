"""Test doubles. The ``LLMClient`` Protocol pays off here: a fake is a few lines
and needs no Ollama daemon.
"""

from __future__ import annotations

from collections.abc import Sequence

from thesis_bench.config import GenerationOptions
from thesis_bench.llm.base import ChatMessage, LLMResponse, LLMUsage


class FakeLLMClient:
    """Returns scripted responses in order; satisfies the ``LLMClient`` Protocol.

    Each scripted item is the text to return. Usage is synthesized with token
    counts derived from text length so accumulation across attempts is testable.
    """

    def __init__(self, responses: Sequence[str], model_id: str = "fake-model") -> None:
        self._responses = list(responses)
        self._model_id = model_id
        self.calls: list[list[ChatMessage]] = []
        self._index = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def chat(
        self, messages: list[ChatMessage], options: GenerationOptions
    ) -> LLMResponse:
        self.calls.append(messages)
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        usage = LLMUsage(
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
            ttft_s=0.01,
            total_generation_s=0.05,
            load_duration_s=0.0,
            prompt_eval_s=0.02,
            eval_s=0.03,
        )
        return LLMResponse(text=text, usage=usage, stop_reason="stop", error=None)
