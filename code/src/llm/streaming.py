"""Backend-agnostic consumption of a streamed chat completion.

Both backends produce the same logical event stream — content deltas, an optional
finish_reason, an optional usage block — and need identical bookkeeping around it: TTFT
stamping, think-block tracking, and the final metrics assembly. That bookkeeping lives here
once; each backend only translates its transport (llama.cpp chunks, SSE lines) into
:class:`StreamEvent` values.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from llm.thinking import ThinkTracker
from llm.types import LLMMetrics, LLMResponse, derive_tokens_per_second, empty_metrics


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One normalized chunk of a streamed completion."""

    content: str | None = None  # text delta, if any
    finish_reason: str | None = None  # set on the final content chunk
    usage_input_tokens: int | None = None  # server-reported prompt tokens, if provided
    usage_output_tokens: int | None = None  # server-reported completion tokens, if provided


def consume_stream(
    events: Iterable[StreamEvent],
    *,
    start: float,
    fallback_input_tokens: Callable[[], int],
    fallback_output_tokens: Callable[[str], int],
) -> LLMResponse:
    """Drain ``events`` into a normalized :class:`LLMResponse`.

    ``start`` is the perf_counter timestamp of request submission (TTFT is measured against
    it). The fallbacks supply token counts when the stream carries no usage block: the input
    fallback is called with no arguments, the output fallback with the full raw text.

    Never raises: any exception from the event iterator (transport errors, malformed chunks)
    is captured in ``metrics["error"]``, honoring the backend contract.
    """
    metrics: LLMMetrics = empty_metrics()
    tracker = ThinkTracker()
    usage_in: int | None = None
    usage_out: int | None = None

    try:
        first_token_seen = False
        for event in events:
            if event.content:
                if not first_token_seen:
                    metrics["time_to_first_token_s"] = time.perf_counter() - start
                    first_token_seen = True
                tracker.feed(event.content)
            if event.finish_reason:
                metrics["stop_reason"] = event.finish_reason
            if event.usage_input_tokens is not None:
                usage_in = event.usage_input_tokens
            if event.usage_output_tokens is not None:
                usage_out = event.usage_output_tokens

        metrics["total_generation_time_s"] = time.perf_counter() - start
        metrics["reasoning_time_s"] = tracker.reasoning_time_s()
        metrics["input_tokens"] = usage_in if usage_in is not None else fallback_input_tokens()
        # Token accounting covers the FULL generation, think block included — reasoning
        # tokens are paid tokens; only the visible text is the usable answer.
        metrics["output_tokens"] = (
            usage_out if usage_out is not None else fallback_output_tokens(tracker.raw_text())
        )
        metrics["tokens_per_second"] = derive_tokens_per_second(metrics)
        return LLMResponse(text=tracker.visible_text(), metrics=metrics)

    except Exception as exc:  # never raise to the caller (backend contract)
        metrics["error"] = str(exc)
        metrics["total_generation_time_s"] = time.perf_counter() - start
        return LLMResponse(text=tracker.visible_text(), metrics=metrics)
