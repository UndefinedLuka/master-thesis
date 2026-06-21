"""Accuracy metrics and the canonical result-set → text conversion.

``rows_to_text`` is used for BOTH the reference answer and the Text-to-SQL
output so the two are compared on a level field (identical formatting on each
side; only the content differs). ROUGE-L is cheap and computed inline during a
run; BERTScore loads a transformer and is computed in a single batched pass in
the aggregation step (see ``cli/aggregate.py``) to keep the GPU free for vLLM.
"""

from __future__ import annotations

from typing import Any, Iterable

# rouge_score is pure-python and cheap; import lazily-safe at module load.
from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _cell_to_text(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        # Stable, compact numeric formatting so 1.0 and 1 do not diverge.
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def rows_to_text(rows: list[dict] | None) -> str:
    """Deterministically render a result set to text.

    Rows are rendered as tab-separated values, one row per line, with the row
    list sorted so that order-insensitive result sets compare equal as text.
    """
    if not rows:
        return ""
    rendered = ["\t".join(_cell_to_text(v) for v in row.values()) for row in rows]
    rendered.sort()
    return "\n".join(rendered)


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0
    return float(_ROUGE.score(reference, hypothesis)["rougeL"].fmeasure)


def _normalize_row(row: dict) -> tuple:
    """Order-insensitive, label-insensitive normalized value tuple for a row."""
    return tuple(_cell_to_text(v) for v in row.values())


def result_set_match(
    pred_rows: list[dict] | None,
    gold_rows: list[dict] | None,
    order_sensitive: bool = False,
) -> bool:
    """Compare two result sets as multisets of normalized value tuples.

    Column names/order are ignored (the model may alias or reorder columns).
    """
    if pred_rows is None or gold_rows is None:
        return False
    pred = [_normalize_row(r) for r in pred_rows]
    gold = [_normalize_row(r) for r in gold_rows]
    if order_sensitive:
        return pred == gold
    return sorted(pred) == sorted(gold)


def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str | None = None,
    batch_size: int = 64,
) -> list[float]:
    """Batched BERTScore F1 for parallel hypothesis/reference lists.

    Imported lazily because it pulls in torch/transformers. Returns one F1 per
    pair; empty-vs-empty scores 1.0, empty-vs-nonempty scores 0.0 (skipped from
    the model call).
    """
    from bert_score import score as bert_score_fn  # lazy: heavy import

    out: list[float] = [0.0] * len(hypotheses)
    idx: list[int] = []
    hyp: list[str] = []
    ref: list[str] = []
    for i, (h, r) in enumerate(zip(hypotheses, references)):
        if not h and not r:
            out[i] = 1.0
        elif not h or not r:
            out[i] = 0.0
        else:
            idx.append(i)
            hyp.append(h)
            ref.append(r)

    if not hyp:
        return out

    kwargs: dict[str, Any] = {"lang": "en", "batch_size": batch_size, "rescale_with_baseline": False}
    if model_type:
        kwargs.pop("lang")
        kwargs["model_type"] = model_type
    _p, _r, f1 = bert_score_fn(hyp, ref, **kwargs)
    for pos, score in zip(idx, f1.tolist()):
        out[pos] = float(score)
    return out


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
