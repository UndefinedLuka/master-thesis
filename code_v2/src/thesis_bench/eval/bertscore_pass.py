"""Optional BERTScore post-pass over result NDJSON files.

BERTScore uses contextual embeddings to reward semantically-equivalent answers
that ROUGE-L would penalize (synonyms, paraphrases). It pulls torch +
transformers and downloads a model, so it is isolated here and run *after* the
benchmark — never in the hot loop. ``bert_score`` is imported lazily so this
module (and the rest of the package) imports fine without the optional extra.

The pass reads each record's ``result.answer_text`` and ``reference_text``,
scores them all in one batch (BERTScore amortizes model load horribly per-call),
and writes ``scores.bertscore_f1`` back into the records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# A small, widely-available model keeps the download light and cross-platform.
DEFAULT_MODEL_TYPE = "roberta-large"


def _compute_f1(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str,
    device: str | None,
) -> list[float]:
    """Batch-score (prediction, reference) pairs, returning F1 per pair.

    ``bert_score`` (and torch) ship no type stubs and are optional, so the call
    is contained behind an ``Any`` boundary and its output coerced to plain
    floats before leaving this function.
    """
    from bert_score import score as bert_score_fn  # lazy: optional dependency

    result: Any = bert_score_fn(
        cands=predictions,
        refs=references,
        model_type=model_type,
        lang="en",
        device=device,
        verbose=False,
    )
    f1: Any = result[2]  # (precision, recall, f1) tensors
    return [float(x) for x in f1.tolist()]


def score_file(
    path: Path,
    *,
    model_type: str = DEFAULT_MODEL_TYPE,
    device: str | None = None,
) -> int:
    """Add ``bertscore_f1`` to every record in one NDJSON file in place.

    Returns the number of records scored. Records whose answer or reference is
    empty are scored 0.0 without invoking the model.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    pairs: list[tuple[int, str, str]] = []
    for idx, rec in enumerate(records):
        pred = str(rec.get("result", {}).get("answer_text", ""))
        ref = str(rec.get("reference_text", ""))
        if pred.strip() and ref.strip():
            pairs.append((idx, pred, ref))
        else:
            rec.setdefault("scores", {})["bertscore_f1"] = 0.0

    if pairs:
        f1s = _compute_f1(
            [p for _, p, _ in pairs],
            [r for _, _, r in pairs],
            model_type=model_type,
            device=device,
        )
        for (idx, _, _), f1 in zip(pairs, f1s, strict=True):
            records[idx].setdefault("scores", {})["bertscore_f1"] = f1

    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)
