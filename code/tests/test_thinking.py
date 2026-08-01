"""Tests for reasoning-time tracking (<think> block detection and stripping)."""

from __future__ import annotations

from typing import Any

from llm.backends.llama_cpp_backend import LlamaCppBackend
from llm.thinking import ThinkTracker
from llm.types import LLMRequest


def _feed_all(tracker: ThinkTracker, deltas: list[str]) -> ThinkTracker:
    for delta in deltas:
        tracker.feed(delta)
    return tracker


def test_no_think_block_passes_through() -> None:
    tracker = _feed_all(ThinkTracker(), ["The answer ", "is 5."])
    assert tracker.visible_text() == "The answer is 5."
    assert tracker.raw_text() == "The answer is 5."
    assert tracker.reasoning_time_s() == 0.0


def test_think_block_is_timed_and_stripped() -> None:
    tracker = _feed_all(
        ThinkTracker(), ["<think>", "hmm, singers", "</think>", "\nThe answer is 5."]
    )
    assert tracker.visible_text() == "The answer is 5."
    assert "<think>" in tracker.raw_text()  # token accounting sees the full text
    assert tracker.reasoning_time_s() > 0.0
    assert tracker.reasoning_time_s() <= 1.0  # sanity: wall-clock of a few feed() calls


def test_close_tag_split_across_deltas() -> None:
    tracker = _feed_all(ThinkTracker(), ["<think>reasoning</th", "ink>answer"])
    assert tracker.visible_text() == "answer"
    assert tracker.reasoning_time_s() > 0.0


def test_close_tag_without_open_tag() -> None:
    # Some chat templates open the think block themselves, so the stream starts mid-think.
    tracker = _feed_all(ThinkTracker(), ["already thinking...", "</think>", "answer"])
    assert tracker.visible_text() == "answer"
    assert tracker.reasoning_time_s() > 0.0


def test_truncated_think_block_is_stripped() -> None:
    # Generation hit max_tokens mid-think: no closing tag ever arrives. The unfinished
    # monologue must not leak into the scored answer.
    tracker = _feed_all(ThinkTracker(), ["<think>step 1... step 2...", " still going"])
    assert tracker.visible_text() == ""
    assert "<think>" in tracker.raw_text()  # token accounting still sees everything


def test_truncated_think_block_keeps_text_before_open_tag() -> None:
    tracker = _feed_all(ThinkTracker(), ["Partial answer. <think>unfinished reasoning"])
    assert tracker.visible_text() == "Partial answer."


def test_llama_backend_reports_reasoning_time() -> None:
    class ThinkingLlama:
        _ctx = None

        def reset(self) -> None:
            pass

        def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False):
            return list(range(len(data.split())))

        def create_chat_completion(self, **kwargs: Any):
            # think block split across deltas, as a real token stream delivers it
            yield {"choices": [{"delta": {"content": "<think>let me see"}}]}
            yield {"choices": [{"delta": {"content": "</think>"}}]}
            yield {"choices": [{"delta": {"content": "42"}, "finish_reason": None}]}
            yield {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 8},
            }

    backend = LlamaCppBackend(llm=ThinkingLlama(), model_id="m", n_ctx=1024)
    response = backend.generate(LLMRequest(messages=[{"role": "user", "content": "q"}]))
    assert response.text == "42"
    assert response.metrics["reasoning_time_s"] > 0.0
    assert response.metrics["output_tokens"] == 8  # think tokens still counted
