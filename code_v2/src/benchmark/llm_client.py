"""OpenAI-compatible streaming client for the vLLM endpoint.

This is the core operational-metrics component. Key correctness points:

* **Real token counts** come from the API ``usage`` object (requires
  ``stream_options={"include_usage": True}``) — never ``len(text)``.
* **TTFT** is measured at the first chunk that carries actual *content*, not the
  role-only opening delta some servers emit.
* The final usage-only chunk has an **empty ``choices`` list** — indexing it
  blindly (as the prototype did) raises ``IndexError``; we guard against it.
* All errors are captured into :class:`GenResult.error`, never raised, so a
  single bad question cannot abort a multi-hour run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from openai import OpenAI


@dataclass
class GenResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    total_generation_s: float = 0.0
    finish_reason: str | None = None
    error: str | None = None


class LLMClient:
    """Thin wrapper around the OpenAI client pointed at a vLLM server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)

    def health(self) -> bool:
        """Return True if the server answers ``GET /health`` or lists models."""
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        for url in (f"{root}/health", f"{self.base_url}/models"):
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    return True
            except requests.RequestException:
                continue
        return False

    def generate(self, system: str, user: str) -> GenResult:
        """Stream a chat completion and collect text + operational metrics."""
        result = GenResult()
        start = time.perf_counter()
        first_content_seen = False
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
                stream_options={"include_usage": True},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            for chunk in stream:
                # The trailing usage-only chunk carries no choices.
                choices = chunk.choices or []
                if choices:
                    delta = choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        if not first_content_seen:
                            result.ttft_s = time.perf_counter() - start
                            first_content_seen = True
                        result.text += content
                    if choices[0].finish_reason:
                        result.finish_reason = choices[0].finish_reason

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    result.input_tokens = usage.prompt_tokens or 0
                    result.output_tokens = usage.completion_tokens or 0

            result.total_generation_s = time.perf_counter() - start
        except Exception as e:  # noqa: BLE001 - capture, never abort the run
            result.error = str(e)
            result.total_generation_s = time.perf_counter() - start

        return result
