"""BERTScore via the ``bert-score`` package (torch + transformers).

The scorer model is loaded once and cached per ``(model_type, lang)`` so a whole batch reuses
a single in-memory model — instantiating per call is the main performance trap. Heavy imports
are lazy so the package imports without the optional ``[metrics]`` extra.

Defaults are pinned (``roberta-large`` / ``en``) for reproducibility; change them in
``config/config.py`` (BERTSCORE_MODEL / BERTSCORE_LANG). Scores are the raw BERTScore values:
baseline rescaling is disabled, so values are not adjusted to spread across a human-readable range.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from config.config import BERTSCORE_LANG as DEFAULT_LANG
from config.config import BERTSCORE_MODEL as DEFAULT_MODEL_TYPE

if TYPE_CHECKING:
    from bert_score import BERTScorer

ScoreTriple = tuple[float, float, float]  # (precision, recall, f1)


@lru_cache(maxsize=4)
def _scorer(model_type: str, lang: str) -> BERTScorer:
    try:
        from bert_score import BERTScorer
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "BERTScore requires the optional 'metrics' dependencies. "
            "Install them with: uv sync --extra metrics"
        ) from exc
    return BERTScorer(
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
    )


def bert_score_batch(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = DEFAULT_MODEL_TYPE,
    lang: str = DEFAULT_LANG,
) -> list[ScoreTriple]:
    """Batched BERTScore — the preferred entry point (one model load, many pairs).

    Returns one ``(precision, recall, f1)`` triple per input pair, in order.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) must align"
        )
    if not predictions:
        return []
    # bert_score's sent_encode branches on "" or " " and calls
    # tokenizer.build_inputs_with_special_tokens([]) which was removed in
    # newer transformers. Use "." so empty inputs skip that branch.
    predictions = [p if p.strip() else "." for p in predictions]
    references = [r if r.strip() else "." for r in references]
    precision, recall, f1 = _scorer(model_type, lang).score(predictions, references)
    return [
        (float(p), float(r), float(f))
        for p, r, f in zip(precision.tolist(), recall.tolist(), f1.tolist(), strict=True)
    ]
