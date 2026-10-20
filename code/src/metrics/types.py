"""Scoring result types (shared by the deferred metrics implementations and pipelines)."""

from __future__ import annotations

from typing import TypedDict


class AccuracyScores(TypedDict):
    """Retrieval-accuracy scores attached to a pipeline result by the scoring pass.

    BERTScore precision, recall, and F1 contain raw, unrescaled similarities when computed.
    Failed outputs with a usable reference receive zero for each enabled accuracy metric; these
    BERTScore fields are ``None`` instead when scoring ran with ``--no-bertscore``. Records
    without a usable reference are not scored. Thus ``None`` means "not computed," distinct from
    a computed raw score or an explicit failure zero.
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
