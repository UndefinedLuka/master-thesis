"""Direct Context Inference pipeline.

The (sampled) database is serialized into the configured format, embedded in the
system prompt, and the model answers the question directly. A single LLM call;
the answer text is the model's raw output.
"""

from __future__ import annotations

from ..enums import PipelineKind
from ..llm.base import ChatMessage
from ..prompts.loader import PromptTemplate, build_prompt
from ..serialization import registry
from .base import Pipeline, PipelineContext
from .types import PipelineResult


class DirectPipeline(Pipeline):
    kind = PipelineKind.DIRECT

    def execute(self, ctx: PipelineContext) -> PipelineResult:
        data = registry.serialize(ctx.fmt, ctx.snapshot)
        system_prompt = build_prompt(PromptTemplate.DIRECT_ANSWER, data=data)

        response = ctx.client.chat(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=ctx.item.question),
            ],
            options=ctx.options,
        )

        # For the direct pipeline, total latency is just the model time; there is
        # no query execution to fold in.
        return PipelineResult(
            answer_text=response.text.strip(),
            raw_model_output=response.text,
            total_usage=response.usage,
            total_latency_s=response.usage.total_generation_s,
            execution_ok=None,
            sql_outcome=None,
            error=response.error,
        )
