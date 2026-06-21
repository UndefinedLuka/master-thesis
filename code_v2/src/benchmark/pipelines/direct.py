"""Direct Context Inference pipeline.

The serialized database (read from the pre-built ``processed/`` file) is placed
in the system prompt; the model answers the question in free-form natural
language; the answer is scored against the reference with ROUGE-L inline.
"""

from __future__ import annotations

from benchmark.datasets import Question
from benchmark.llm_client import LLMClient
from benchmark.metrics import compute_rouge_l
from benchmark.prompts import build_direct_prompt
from benchmark.reference import Reference
from benchmark.results import ResultRow


def run_direct_question(
    client: LLMClient,
    model_name: str,
    dataset: str,
    fmt: str,
    q: Question,
    serialized: str,
    ref: Reference | None,
) -> ResultRow:
    system = build_direct_prompt(serialized, q.evidence)
    gen = client.generate(system, q.question)

    reference_text = ref.answer_text if ref else ""
    reference_unavailable = ref is None or ref.error is not None

    rouge = 0.0
    if not reference_unavailable and gen.error is None:
        rouge = compute_rouge_l(gen.text, reference_text)

    return ResultRow(
        model=model_name,
        dataset=dataset,
        pipeline="direct",
        format=fmt,
        question_index=q.index,
        db_id=q.db_id,
        question=q.question,
        output_text=gen.text,
        reference_text=reference_text,
        reference_unavailable=reference_unavailable,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        ttft_s=gen.ttft_s,
        total_generation_s=gen.total_generation_s,
        rouge_l=rouge,
        error=gen.error,
    )
