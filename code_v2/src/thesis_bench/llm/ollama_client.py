"""Ollama-backed ``LLMClient``.

This is where v1's three metric bugs are fixed:

1. **Output tokens** come from Ollama's ``eval_count`` (real generated tokens),
   not ``len(text)`` (characters).
2. **Input tokens** come from ``prompt_eval_count`` (what the model actually
   ingested), not a separate re-tokenization of the prompt string.
3. **TTFT** is measured client-side at the first content-bearing stream chunk and
   ``load_duration`` is tracked separately, so the one-time model load never
   inflates time-to-first-token.

We stream so TTFT is real wall-clock; the final chunk (``done=True``) carries the
native counters, which we read off it.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from ollama import Client

from ..config import GenerationOptions
from ..errors import LLMError
from .base import ChatMessage, LLMResponse, LLMUsage

# Nanoseconds (Ollama's duration unit) per second.
_NS_PER_S = 1_000_000_000.0


def _ns_to_s(value: Any) -> float:
    """Convert an Ollama nanosecond duration (possibly absent/None) to seconds."""
    if value is None:
        return 0.0
    try:
        return float(value) / _NS_PER_S
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _options_payload(options: GenerationOptions) -> dict[str, Any]:
    """Translate typed ``GenerationOptions`` into Ollama's ``options`` dict.

    This typed-to-dict boundary is the only place option names are spelled out,
    so a misspelling is caught here rather than silently ignored by the server.
    """
    payload: dict[str, Any] = {
        "temperature": options.temperature,
        "seed": options.seed,
        "num_ctx": options.num_ctx,
        "num_predict": options.num_predict,
        "top_p": options.top_p,
    }
    if options.stop:
        payload["stop"] = list(options.stop)
    return payload


class OllamaClient:
    """Default backend. Talks HTTP to a local Ollama daemon (OS-agnostic)."""

    def __init__(self, model_id: str, host: str | None = None) -> None:
        self._model_id = model_id
        # ``host=None`` lets the client honor the OLLAMA_HOST env var / default.
        self._client = Client(host=host)

    @property
    def model_id(self) -> str:
        return self._model_id

    def list_local_models(self) -> list[str]:
        """Model tags currently available in the daemon (for preflight checks)."""
        try:
            listing = self._client.list()
        except Exception as exc:  # noqa: BLE001 - normalize transport errors
            raise LLMError(f"Could not reach Ollama daemon: {exc}") from exc
        names: list[str] = []
        for model in listing.models:
            name = getattr(model, "model", None)
            if name:
                names.append(str(name))
        return names

    def chat(
        self, messages: list[ChatMessage], options: GenerationOptions
    ) -> LLMResponse:
        wire_messages = [{"role": m.role, "content": m.content} for m in messages]
        start = time.perf_counter()

        try:
            # ``Client.chat`` is overloaded and only partially typed; route through
            # an ``Any`` iterator so the untyped surface stays contained here.
            stream: Iterable[Any] = self._client.chat(
                model=self._model_id,
                messages=wire_messages,
                options=_options_payload(options),
                stream=True,
            )

            text_parts: list[str] = []
            ttft_s = 0.0
            stop_reason: str | None = None
            final_chunk: Any = None

            for chunk in stream:
                content = chunk.message.content if chunk.message else None
                if content:
                    if ttft_s == 0.0:
                        ttft_s = time.perf_counter() - start
                    text_parts.append(content)
                if chunk.done:
                    stop_reason = chunk.done_reason
                    final_chunk = chunk

        except Exception as exc:  # noqa: BLE001 - any backend failure -> recorded
            return LLMResponse(
                text="",
                usage=LLMUsage.zero(),
                stop_reason=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        usage = self._usage_from_chunk(final_chunk, ttft_s)
        return LLMResponse(
            text="".join(text_parts),
            usage=usage,
            stop_reason=stop_reason,
            error=None,
        )

    @staticmethod
    def _usage_from_chunk(final_chunk: Any, ttft_s: float) -> LLMUsage:
        """Read native counters/timings off the final (``done``) stream chunk."""
        if final_chunk is None:
            return LLMUsage(0, 0, ttft_s, 0.0, 0.0, 0.0, 0.0)

        prompt_eval_s = _ns_to_s(getattr(final_chunk, "prompt_eval_duration", None))
        eval_s = _ns_to_s(getattr(final_chunk, "eval_duration", None))
        return LLMUsage(
            input_tokens=_to_int(getattr(final_chunk, "prompt_eval_count", None)),
            output_tokens=_to_int(getattr(final_chunk, "eval_count", None)),
            ttft_s=ttft_s,
            total_generation_s=prompt_eval_s + eval_s,
            load_duration_s=_ns_to_s(getattr(final_chunk, "load_duration", None)),
            prompt_eval_s=prompt_eval_s,
            eval_s=eval_s,
        )
