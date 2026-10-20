"""Scoring orchestration: turn benchmark records + references into scored records.

Pure and I/O-free. The metric functions are injected (defaulting to the real implementations)
so the orchestration can be unit-tested without torch/transformers. BERTScore is computed once
over the whole batch — the expensive model loads a single time.

Text construction (Chapter 3 §Retrieval Accuracy):
  * Direct Context Inference  -> prediction text is the model's raw NL answer.
  * Text-to-SQL               -> prediction text is the executed result rows, stringified.
  * Reference text (both)     -> the gold result rows, stringified.
  * Empty Direct references   -> the prompt's canonical no-answer text, ``N/A``.

Text-to-SQL additionally gets binary Execution Accuracy (whether its query ran successfully).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from metrics.bertscore import (
    DEFAULT_LANG,
    DEFAULT_MODEL_TYPE,
    ScoreTriple,
    bert_score_batch,
)
from metrics.exact_match import normalized_exact_match
from metrics.execution import execution_accuracy
from metrics.rouge import rouge_l
from metrics.types import AccuracyScores
from pipelines.types import PipelineResult

# A reference row set keyed by example_id: gold rows (or None if gold SQL errored).
ReferenceLookup = dict[str, "ReferenceEntry"]

ExactMatchFn = Callable[[str, str], float]
RougeFn = Callable[[str, str], float]
BertBatchFn = Callable[[list[str], list[str]], list[ScoreTriple]]

DIRECT_NO_ANSWER_TEXT = "N/A"


class ReferenceEntry(TypedDict):
    """One example's gold result rows (None if the gold SQL errored)."""

    rows: list[dict[str, Any]] | None


def rows_to_text(rows: list[dict[str, Any]] | None) -> str:
    """Stringify result rows into a stable, compact text for text-similarity metrics.

    One row per line, cells joined by ``" | "`` in column order. ``None`` cells render empty.
    Empty/absent result sets render as the empty string.
    """
    if not rows:
        return ""
    lines = [
        " | ".join("" if value is None else str(value) for value in row.values()) for row in rows
    ]
    return "\n".join(lines)


def prediction_text(result: PipelineResult) -> str:
    """The text a result is scored on: raw answer for direct, stringified rows for SQL."""
    if result["pipeline"] == "text_to_sql":
        return rows_to_text(result["predicted_rows"])
    return result["raw_output"]


def reference_text(entry: ReferenceEntry) -> str:
    """The gold text both pipelines are scored against (stringified gold rows)."""
    return rows_to_text(entry["rows"])


def scoring_text_pair(result: PipelineResult, entry: ReferenceEntry) -> tuple[str, str]:
    """Build comparable prediction/reference text for one successful pipeline result.

    Direct inference is explicitly prompted to return ``N/A`` when the provided data contains
    no answer. Gold SQL represents that same condition as empty rendered text, so use the prompt's
    sentinel as the Direct reference. Keeping this rule pair-specific preserves the existing empty
    result representation used by Text-to-SQL.
    """
    prediction = prediction_text(result)
    reference = reference_text(entry)
    if result["pipeline"] == "direct" and not reference.strip():
        reference = DIRECT_NO_ANSWER_TEXT
    return prediction, reference


class ScoringStats(TypedDict):
    """Summary counters returned by :func:`score_records` alongside the scored copies."""

    total: int
    scored: int  # got computed scores or explicit zeroes for a failed output
    failed_output_zeroed: int  # valid reference, but pipeline/backend reported an error
    skipped_no_reference: int  # no matching reference, or gold SQL errored
    execution_scored: int  # text_to_sql records given an execution verdict
    bertscore_enabled: bool


def score_records(
    records: list[PipelineResult],
    references: ReferenceLookup,
    *,
    with_bertscore: bool = True,
    exact_match_fn: ExactMatchFn = normalized_exact_match,
    rouge_fn: RougeFn = rouge_l,
    bert_batch_fn: BertBatchFn | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    lang: str = DEFAULT_LANG,
) -> tuple[list[PipelineResult], ScoringStats]:
    """Return scored copies of ``records`` plus summary stats.

    A record is text-scored only when a reference with non-null gold rows exists for its
    ``example_id``. If such a record has a pipeline/backend ``error``, all enabled accuracy
    metrics are set to zero without comparing its placeholder output to the reference. This
    prevents failed empty outputs from matching empty gold result sets. Text-to-SQL records
    additionally receive binary Execution Accuracy based only on whether the query ran. Heavy
    BERTScore runs once over the eligible, non-failed batch.

    ``exact_match_fn`` / ``rouge_fn`` / ``bert_batch_fn`` are injectable for testing; all
    default to the real implementations.
    """
    scored: list[PipelineResult] = [dict(record) for record in records]  # type: ignore[misc]

    # Build the eligible-for-text-scoring set and their aligned prediction/reference texts.
    eligible_idx: list[int] = []
    predictions: list[str] = []
    refs: list[str] = []

    stats_failed = 0
    stats_skipped = 0
    stats_exec = 0

    for idx, record in enumerate(scored):
        entry = references.get(record["example_id"])

        # Binary execution success (text_to_sql only), independent of text-scoring eligibility.
        if record["pipeline"] == "text_to_sql":
            record["execution"] = execution_accuracy(record["predicted_rows"])
            stats_exec += 1

        # Accuracy scoring requires a usable gold reference. A failed output with a valid
        # reference receives explicit zeroes instead of comparing placeholder/empty content.
        if entry is None or entry["rows"] is None:
            stats_skipped += 1
            continue
        if _output_failed(record):
            record["scores"] = _failed_output_scores(with_bertscore=with_bertscore)
            stats_failed += 1
            continue
        eligible_idx.append(idx)
        prediction, reference = scoring_text_pair(record, entry)
        predictions.append(prediction)
        refs.append(reference)

    # Normalized Exact Match and ROUGE-L per pair; BERTScore in one batch.
    pairs = zip(predictions, refs, strict=True)
    exact_match_scores = [exact_match_fn(pred, ref) for pred, ref in pairs]
    rouge_scores = [rouge_fn(pred, ref) for pred, ref in zip(predictions, refs, strict=True)]

    # None (not 0.0) when BERTScore is disabled, so "not computed" stays distinguishable.
    bert_scores: list[ScoreTriple] | list[tuple[None, None, None]]
    if with_bertscore and predictions:
        batch_fn = bert_batch_fn or _default_bert_batch(model_type, lang)
        bert_scores = batch_fn(predictions, refs)
    else:
        bert_scores = [(None, None, None)] * len(predictions)

    for offset, idx in enumerate(eligible_idx):
        p, r, f = bert_scores[offset]
        scored[idx]["scores"] = AccuracyScores(
            normalized_exact_match=exact_match_scores[offset],
            rouge_l_f=rouge_scores[offset],
            bertscore_p=p,
            bertscore_r=r,
            bertscore_f=f,
        )

    stats: ScoringStats = {
        "total": len(records),
        "scored": len(eligible_idx) + stats_failed,
        "failed_output_zeroed": stats_failed,
        "skipped_no_reference": stats_skipped,
        "execution_scored": stats_exec,
        "bertscore_enabled": with_bertscore,
    }
    return scored, stats


def _output_failed(record: PipelineResult) -> bool:
    """Whether a record has no valid answer that should enter text-similarity metrics."""
    if record["error"] is not None:
        return True
    return record["pipeline"] == "text_to_sql" and record["execution_success"] is False


def _failed_output_scores(*, with_bertscore: bool) -> AccuracyScores:
    """Zero accuracy for a failed output without running text similarity metrics."""
    bertscore_value = 0.0 if with_bertscore else None
    return AccuracyScores(
        normalized_exact_match=0.0,
        rouge_l_f=0.0,
        bertscore_p=bertscore_value,
        bertscore_r=bertscore_value,
        bertscore_f=bertscore_value,
    )


def references_from_rows(reference_rows: Sequence[dict[str, Any]]) -> ReferenceLookup:
    """Index loaded ReferenceAnswer dicts by ``example_id`` for fast lookup during scoring."""
    return {row["example_id"]: ReferenceEntry(rows=row.get("rows")) for row in reference_rows}


def _default_bert_batch(model_type: str, lang: str) -> BertBatchFn:
    def run(predictions: list[str], refs: list[str]) -> list[ScoreTriple]:
        return bert_score_batch(predictions, refs, model_type=model_type, lang=lang)

    return run
