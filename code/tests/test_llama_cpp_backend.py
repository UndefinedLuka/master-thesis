"""Tests for the llama_cpp backend's no-context-leak guarantee and config wiring.

A fake ``Llama`` records the order of state-clearing vs generation so we can assert that the
KV cache is wiped before every question — without loading a real model.
"""

from __future__ import annotations

from typing import Any

from llm.backends.llama_cpp_backend import LlamaCppBackend
from llm.types import LLMRequest


class FakeCtx:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def kv_cache_clear(self) -> None:
        self._log.append("kv_cache_clear")


class FakeLlama:
    """Minimal stand-in for llama_cpp.Llama that logs state-mutating calls."""

    n_tokens = 7  # prompt tokens "evaluated" when the stream starts

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._ctx = FakeCtx(log)

    def reset(self) -> None:
        self._log.append("reset")

    def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False) -> list[int]:
        return list(range(len(data.split())))

    def create_chat_completion(self, **kwargs: Any):
        self._log.append("generate")
        # Two streamed chunks: one content token, then a final chunk with finish_reason + usage.
        yield {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
        yield {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }


def _backend(log: list[str]) -> LlamaCppBackend:
    return LlamaCppBackend(llm=FakeLlama(log), model_id="fake.gguf", n_ctx=1024)


def _request() -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "one two three"}])


def test_state_cleared_before_generation() -> None:
    log: list[str] = []
    backend = _backend(log)
    backend.generate(_request())
    # KV cache cleared AND token counter reset, both before generation begins.
    assert log.index("kv_cache_clear") < log.index("generate")
    assert log.index("reset") < log.index("generate")


def test_no_leak_across_two_questions() -> None:
    log: list[str] = []
    backend = _backend(log)
    backend.generate(_request())
    backend.generate(_request())
    # Each of the two generations is preceded by its own clear+reset (no shared/leaked state).
    assert log == [
        "kv_cache_clear",
        "reset",
        "generate",
        "kv_cache_clear",
        "reset",
        "generate",
    ]


def test_generate_reports_token_usage_not_char_count() -> None:
    backend = _backend([])
    response = backend.generate(_request())
    assert response.text == "hi"
    assert response.metrics["output_tokens"] == 1  # from usage block, not len("hi")
    assert response.metrics["stop_reason"] == "stop"
    assert response.metrics["error"] is None


def test_input_tokens_from_model_counter() -> None:
    # The prompt-token count comes from the model's n_tokens at the first streamed chunk
    # (includes chat-template tokens), not from re-tokenizing the message contents.
    backend = _backend([])
    response = backend.generate(_request())
    assert response.metrics["input_tokens"] == FakeLlama.n_tokens


def test_request_carries_generation_params_to_backend() -> None:
    # max_tokens / temperature / seed must reach the model call.
    captured: dict[str, object] = {}

    class CapturingLlama(FakeLlama):
        def create_chat_completion(self, **kwargs: Any):
            captured.update(kwargs)
            return super().create_chat_completion(**kwargs)

    backend = LlamaCppBackend(llm=CapturingLlama([]), model_id="m", n_ctx=1024)
    backend.generate(
        LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2_048,
            temperature=0.7,
            seed=42,
        )
    )
    assert captured["max_tokens"] == 2_048
    assert captured["temperature"] == 0.7
    assert captured["seed"] == 42


def test_generation_error_is_captured_not_raised() -> None:
    class ExplodingLlama(FakeLlama):
        def create_chat_completion(self, **kwargs: Any):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this a generator

    backend = LlamaCppBackend(llm=ExplodingLlama([]), model_id="m", n_ctx=1024)
    response = backend.generate(_request())
    assert response.metrics["error"] is not None
    assert "boom" in response.metrics["error"]
    assert response.text == ""
