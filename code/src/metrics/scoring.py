"""Scoring orchestration: turn benchmark records + references into scored records.

Pure and I/O-free. The metric functions are injected (defaulting to the real implementations)
so the orchestration can be unit-tested without torch/transformers. BERTScore is computed once
over the whole batch — the expensive model loads a single time.

Text construction (Chapter 3 §Retrieval Accuracy):
  * Direct Context Inference  -> prediction text is the model's raw NL answer.
  * Text-to-SQL               -> prediction text is the executed result rows, stringified.
  * Reference text (both)     -> the gold result rows, stringified.

Text-to-SQL additionally gets binary Execution Accuracy (whether its query ran successfully)
and a separate result-match diagnostic on the raw rows.
"""

from __future__ import annotations

import re
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

# A reference row set keyed by example_id: gold rows (or None if gold SQL errored) + gold SQL.
ReferenceLookup = dict[str, "ReferenceEntry"]

ExactMatchFn = Callable[[str, str], float]
RougeFn = Callable[[str, str], float]
BertBatchFn = Callable[[list[str], list[str]], list[ScoreTriple]]


class ReferenceEntry(TypedDict):
    """One example's gold result: rows (None if the gold SQL errored) + the gold SQL."""

    rows: list[dict[str, Any]] | None
    gold_sql: str


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


def has_order_by(gold_sql: str) -> bool:
    """Whether the gold query imposes an order on its RESULT rows.

    Only an ORDER BY at parenthesis depth 0 counts — one inside a subquery/CTE does not
    constrain the outer result's order, so matching it would force order-sensitive comparison
    spuriously.
    """
    depth = 0
    for match in re.finditer(r"[()]|\border\s+by\b", gold_sql, re.IGNORECASE):
        token = match.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            return True
    return False


class ScoringStats(TypedDict):
    """Summary counters returned by :func:`score_records` alongside the scored copies."""

    total: int
    scored: int  # got text-similarity scores
    skipped_no_reference: int  # no matching reference, or gold SQL errored
    execution_scored: int  # text_to_sql records given an execution verdict
    bertscore_enabled: bool


def score_records(
    records: list[PipelineResult],
    references: ReferenceLookup,
    *,
    with_bertscore: bool = True,
    force_order_sensitive: bool | None = None,
    exact_match_fn: ExactMatchFn = normalized_exact_match,
    rouge_fn: RougeFn = rouge_l,
    bert_batch_fn: BertBatchFn | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    lang: str = DEFAULT_LANG,
) -> tuple[list[PipelineResult], ScoringStats]:
    """Return scored copies of ``records`` plus summary stats.

    A record is text-scored only when a reference with non-null gold rows exists for its
    ``example_id``. Text-to-SQL records additionally receive binary Execution Accuracy based
    only on whether the query ran. Their separate result-match diagnostic is order-sensitive
    iff the gold SQL has ``ORDER BY``, unless ``force_order_sensitive`` overrides it. Heavy
    BERTScore runs once over the whole eligible batch.

    ``exact_match_fn`` / ``rouge_fn`` / ``bert_batch_fn`` are injectable for testing; all
    default to the real implementations.
    """
    scored: list[PipelineResult] = [dict(record) for record in records]  # type: ignore[misc]

    # Build the eligible-for-text-scoring set and their aligned prediction/reference texts.
    eligible_idx: list[int] = []
    predictions: list[str] = []
    refs: list[str] = []

    stats_skipped = 0
    stats_exec = 0

    for idx, record in enumerate(scored):
        entry = references.get(record["example_id"])

        # Binary execution success plus result matching (text_to_sql only); both are
        # independent of text-scoring eligibility.
        if record["pipeline"] == "text_to_sql":
            gold_rows = entry["rows"] if entry else None
            order_sensitive = (
                force_order_sensitive
                if force_order_sensitive is not None
                else (has_order_by(entry["gold_sql"]) if entry else False)
            )
            record["execution"] = execution_accuracy(
                record["predicted_rows"], gold_rows, order_sensitive=order_sensitive
            )
            stats_exec += 1

        # Text-similarity scoring requires a usable gold reference.
        if entry is None or entry["rows"] is None:
            stats_skipped += 1
            continue
        eligible_idx.append(idx)
        predictions.append(prediction_text(record))
        refs.append(reference_text(entry))

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
        "scored": len(eligible_idx),
        "skipped_no_reference": stats_skipped,
        "execution_scored": stats_exec,
        "bertscore_enabled": with_bertscore,
    }
    return scored, stats


def references_from_rows(reference_rows: Sequence[dict[str, Any]]) -> ReferenceLookup:
    """Index loaded ReferenceAnswer dicts by ``example_id`` for fast lookup during scoring."""
    return {
        row["example_id"]: ReferenceEntry(rows=row.get("rows"), gold_sql=row.get("gold_sql", ""))
        for row in reference_rows
    }


def _default_bert_batch(model_type: str, lang: str) -> BertBatchFn:
    def run(predictions: list[str], refs: list[str]) -> list[ScoreTriple]:
        return bert_score_batch(predictions, refs, model_type=model_type, lang=lang)

    return run
