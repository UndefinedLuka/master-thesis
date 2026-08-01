"""OpenAI-compatible HTTP backend.

Drives any server exposing ``POST /v1/chat/completions`` — llama.cpp's bundled ``server``,
Ollama, vLLM, or LM Studio — so the inference engine can be swapped (or moved to a remote
GPU box) by changing only ``base_url`` / ``model_id``. Streams Server-Sent Events to measure
TTFT and requests ``stream_options.include_usage`` so split token counts come from the
server when available.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from config.settings import BackendConfig
from llm.streaming import StreamEvent, consume_stream
from llm.types import BackendInfo, LLMRequest, LLMResponse

_DEFAULT_TIMEOUT_S = 600.0


@dataclass(slots=True)
class OpenAICompatBackend:
    """Talks to a local/remote OpenAI-compatible server. Satisfies ``LLMBackend``."""

    client: httpx.Client
    model_id: str

    @classmethod
    def create(cls, cfg: BackendConfig) -> OpenAICompatBackend:
        if not cfg.base_url:
            raise ValueError(
                "openai_compat backend requires BASE_URL in config/config.py "
                "(e.g. http://localhost:8080/v1)"
            )
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT_S, connect=10.0),
        )
        return cls(client=client, model_id=cfg.model_id)

    def info(self) -> BackendInfo:
        return BackendInfo(name="openai_compat", model_id=self.model_id)

    def count_tokens(self, text: str) -> int:
        """Coarse token estimate: ``max(words, chars / 4)``.

        Remote servers rarely expose a tokenizer over HTTP; the authoritative counts come
        from the response ``usage`` block. This estimate exists for context-fit checks and
        as a fallback when usage is absent. A pure word count would badly undercount dense
        serialized data (JSON/CSV with long unbroken tokens), so the character heuristic
        (~4 chars per token for English-like text) provides a floor.
        """
        return max(len(text.split()), len(text) // 4)

    def generate(self, request: LLMRequest) -> LLMResponse:
        prompt_text = "\n".join(m["content"] for m in request.messages)
        start = time.perf_counter()
        return consume_stream(
            self._events(request),
            start=start,
            fallback_input_tokens=lambda: self.count_tokens(prompt_text),
            fallback_output_tokens=self.count_tokens,
        )

    def _events(self, request: LLMRequest) -> Iterator[StreamEvent]:
        """Translate the server's SSE lines into normalized :class:`StreamEvent`s."""
        payload = self._build_payload(request)
        with self.client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                event = _parse_sse_line(line)
                if event is None:
                    continue
                if event == "[DONE]":
                    break
                choices = event.get("choices") or []
                choice = choices[0] if choices else {}
                usage = event.get("usage") or {}
                yield StreamEvent(
                    content=(choice.get("delta") or {}).get("content"),
                    finish_reason=choice.get("finish_reason"),
                    usage_input_tokens=usage.get("prompt_tokens"),
                    usage_output_tokens=usage.get("completion_tokens"),
                )

    def close(self) -> None:
        self.client.close()

    def _build_payload(self, request: LLMRequest) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model_id,
            "messages": request.messages,
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Each question is sent as a full, independent request (servers hold no
            # conversation state between calls). Disable llama.cpp-server's prompt cache so a
            # shared prefix from the previous question can't skew this one's TTFT. Ignored by
            # servers that don't recognize it.
            "cache_prompt": False,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = request.stop
        return payload


def _parse_sse_line(line: str) -> dict[str, Any] | str | None:
    """Decode one SSE line into a JSON object, the ``[DONE]`` sentinel, or None to skip."""
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data:
        return None
    if data == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None
