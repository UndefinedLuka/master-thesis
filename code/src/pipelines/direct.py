"""Direct Context Inference pipeline.

Serialize the whole database into a format, place it in the prompt, ask the question, and
record the model's natural-language answer plus operational-efficiency metrics. Accuracy
scoring is deferred to the separate scoring pass (``scores`` stays ``None`` here).
"""

from __future__ import annotations

from datasets.types import DatasetExample
from llm.backend import LLMBackend
from pipelines.shared import base_result, build_messages, make_request, model_id_of
from pipelines.types import PipelineResult
from prompts import render_prompt
from serialization.types import SerializationFormat


def run_direct_inference(
    backend: LLMBackend,
    example: DatasetExample,
    serialized: str,
    fmt: SerializationFormat,
    prompt_template: str,
    *,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> PipelineResult:
    """Run one Direct Context Inference example end-to-end."""
    result = base_result(example, pipeline="direct", fmt=fmt, model_id=model_id_of(backend))

    system_prompt = render_prompt(prompt_template, serialized)
    messages = build_messages(system_prompt, example["question"], evidence=example["evidence"])
    response = backend.generate(
        make_request(messages, max_tokens=max_tokens, temperature=temperature, seed=seed)
    )

    metrics = response.metrics
    result["raw_output"] = response.text
    result["input_tokens"] = metrics["input_tokens"]
    result["output_tokens"] = metrics["output_tokens"]
    result["time_to_first_token_s"] = metrics["time_to_first_token_s"]
    result["reasoning_time_s"] = metrics["reasoning_time_s"]
    result["generation_time_s"] = metrics["total_generation_time_s"]
    result["total_time_s"] = metrics["total_generation_time_s"]  # no query exec for direct
    result["error"] = metrics["error"]
    return result
