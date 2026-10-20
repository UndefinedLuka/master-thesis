"""Scoring result types (shared by the deferred metrics implementations and pipelines)."""

from __future__ import annotations

from typing import TypedDict


class AccuracyScores(TypedDict):
    """Retrieval-accuracy scores attached to a pipeline result by the scoring pass.

    BERTScore fields are ``None`` when scoring ran with ``--no-bertscore`` — never 0.0,
    so downstream analysis can distinguish "not computed" from "scored zero".
    """

    normalized_exact_match: float
    rouge_l_f: float
    bertscore_p: float | None
    bertscore_r: float | None
    bertscore_f: float | None


class ExecutionScore(TypedDict):
    """Binary Text-to-SQL execution metric."""

    execution_success: bool  # query executed without error (within the retry cap)
    execution_accuracy: float  # 1.0 iff execution_success, otherwise 0.0
