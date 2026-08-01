"""Tests for the OpenAI-compatible HTTP backend (SSE parsing, payload, error contract).

An ``httpx.MockTransport`` stands in for the server, so no network is involved.
"""

from __future__ import annotations

import json

import httpx

from llm.backends.openai_compat_backend import OpenAICompatBackend, _parse_sse_line
from llm.types import LLMRequest


def _sse_body(*events: dict | str) -> str:
    lines = []
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {data}\n\n")
    return "".join(lines)


def _backend_with_response(body: str, status_code: int = 200) -> OpenAICompatBackend:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code, content=body, headers={"Content-Type": "text/event-stream"}
        )

    client = httpx.Client(base_url="http://test/v1", transport=httpx.MockTransport(handler))
    return OpenAICompatBackend(client=client, model_id="m")


def _request(**kwargs) -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "hello world"}], **kwargs)


# --- _parse_sse_line -----------------------------------------------------------------------


def test_parse_sse_line_decodes_json() -> None:
    assert _parse_sse_line('data: {"a": 1}') == {"a": 1}


def test_parse_sse_line_detects_done_sentinel() -> None:
    assert _parse_sse_line("data: [DONE]") == "[DONE]"


def test_parse_sse_line_skips_non_data_and_empty_lines() -> None:
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": keepalive comment") is None
    assert _parse_sse_line("event: ping") is None
    assert _parse_sse_line("data:") is None


def test_parse_sse_line_skips_malformed_json() -> None:
    assert _parse_sse_line("data: {not json") is None


# --- generate ------------------------------------------------------------------------------


def test_generate_collects_text_and_usage() -> None:
    body = _sse_body(
        {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
        "[DONE]",
    )
    response = _backend_with_response(body).generate(_request())
    assert response.text == "Hello"
    assert response.metrics["input_tokens"] == 12
    assert response.metrics["output_tokens"] == 2
    assert response.metrics["stop_reason"] == "stop"
    assert response.metrics["error"] is None
    assert response.metrics["time_to_first_token_s"] > 0.0


def test_generate_falls_back_to_estimate_without_usage() -> None:
    body = _sse_body(
        {"choices": [{"delta": {"content": "one two three"}, "finish_reason": "stop"}]},
        "[DONE]",
    )
    backend = _backend_with_response(body)
    response = backend.generate(_request())
    assert response.metrics["output_tokens"] == backend.count_tokens("one two three")
    assert response.metrics["input_tokens"] == backend.count_tokens("hello world")


def test_generate_http_error_is_captured_not_raised() -> None:
    response = _backend_with_response("nope", status_code=500).generate(_request())
    assert response.metrics["error"] is not None
    assert "500" in response.metrics["error"]
    assert response.text == ""


def test_generate_strips_think_block() -> None:
    body = _sse_body(
        {"choices": [{"delta": {"content": "<think>hmm"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "</think>"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]},
        "[DONE]",
    )
    response = _backend_with_response(body).generate(_request())
    assert response.text == "42"
    assert response.metrics["reasoning_time_s"] > 0.0


# --- payload -------------------------------------------------------------------------------


def test_payload_disables_prompt_cache_and_requests_usage() -> None:
    backend = _backend_with_response("")
    payload = backend._build_payload(_request())
    assert payload["cache_prompt"] is False  # no TTFT skew from a shared prefix
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in payload  # unset -> omitted, not null


def test_payload_forwards_generation_params() -> None:
    backend = _backend_with_response("")
    payload = backend._build_payload(_request(max_tokens=256, temperature=0.5, seed=7))
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.5
    assert payload["seed"] == 7


# --- count_tokens --------------------------------------------------------------------------


def test_count_tokens_word_floor_for_dense_text() -> None:
    backend = _backend_with_response("")
    # Dense JSON-ish text: few whitespace words but many tokens; chars/4 provides the floor.
    dense = '{"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8}'
    assert backend.count_tokens(dense) >= len(dense) // 4
    # Plain prose: word count dominates.
    assert backend.count_tokens("a b c d") == 4
