"""Tests for the normalized LLM types, the factory, and the shared stream consumer."""

from __future__ import annotations

import time

import pytest

from config.settings import BackendConfig
from llm.factory import get_backend
from llm.streaming import StreamEvent, consume_stream
from llm.types import LLMMetrics, derive_tokens_per_second, empty_metrics


def _metrics(**overrides) -> LLMMetrics:
    metrics = empty_metrics()
    metrics.update(overrides)  # type: ignore[typeddict-item]
    return metrics


def test_tokens_per_second_basic() -> None:
    metrics = _metrics(output_tokens=10, total_generation_time_s=2.5, time_to_first_token_s=0.5)
    assert derive_tokens_per_second(metrics) == 10 / 2.0


def test_tokens_per_second_zero_time_guard() -> None:
    # TTFT == total time (single-token responses) must not divide by zero.
    metrics = _metrics(output_tokens=1, total_generation_time_s=0.5, time_to_first_token_s=0.5)
    assert derive_tokens_per_second(metrics) == 0.0


def test_tokens_per_second_zero_tokens_guard() -> None:
    metrics = _metrics(output_tokens=0, total_generation_time_s=1.0)
    assert derive_tokens_per_second(metrics) == 0.0


def test_factory_rejects_unknown_kind() -> None:
    cfg = BackendConfig(model_id="m")
    object.__setattr__(cfg, "kind", "nope")  # bypass the frozen dataclass for the test
    with pytest.raises(ValueError, match="Unknown backend kind"):
        get_backend(cfg)


# --- consume_stream ------------------------------------------------------------------------


def test_consume_stream_assembles_response() -> None:
    events = [
        StreamEvent(content="a"),
        StreamEvent(content="b", finish_reason="stop"),
        StreamEvent(usage_input_tokens=3, usage_output_tokens=2),
    ]
    response = consume_stream(
        iter(events),
        start=time.perf_counter(),
        fallback_input_tokens=lambda: 99,
        fallback_output_tokens=lambda text: 99,
    )
    assert response.text == "ab"
    assert response.metrics["input_tokens"] == 3  # usage wins over fallback
    assert response.metrics["output_tokens"] == 2
    assert response.metrics["stop_reason"] == "stop"


def test_consume_stream_uses_fallbacks_without_usage() -> None:
    response = consume_stream(
        iter([StreamEvent(content="x y z", finish_reason="stop")]),
        start=time.perf_counter(),
        fallback_input_tokens=lambda: 7,
        fallback_output_tokens=lambda text: len(text.split()),
    )
    assert response.metrics["input_tokens"] == 7
    assert response.metrics["output_tokens"] == 3


def test_consume_stream_captures_iterator_error() -> None:
    def exploding():
        yield StreamEvent(content="partial")
        raise RuntimeError("stream died")

    response = consume_stream(
        exploding(),
        start=time.perf_counter(),
        fallback_input_tokens=lambda: 0,
        fallback_output_tokens=lambda text: 0,
    )
    assert response.metrics["error"] is not None
    assert "stream died" in response.metrics["error"]
    assert response.text == "partial"  # text collected before the failure is preserved
