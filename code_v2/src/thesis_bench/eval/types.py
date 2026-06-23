"""Evaluation result types.

Scores apply to both pipelines. ``rouge_l`` is always computed (cheap, pure
Python). ``bertscore_f1`` is ``None`` unless the optional BERTScore post-pass has
run, so the record schema is stable whether or not torch is installed.
``execution_accuracy`` and ``row_set_match`` are ``None`` for the direct pipeline
(which has no executed query).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalScores:
    """Similarity and correctness scores for one benchmark record."""

    rouge_l: float
    bertscore_f1: float | None = None
    # Text-to-SQL only (None for the direct pipeline):
    execution_accuracy: bool | None = None  # did the final query execute?
    row_set_match: bool | None = None  # do executed rows match the reference rows?
