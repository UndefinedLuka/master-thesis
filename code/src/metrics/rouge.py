"""ROUGE-L scoring via ``rouge-score`` (pure-Python, cross-platform).

Reports the ROUGE-L F-measure with Porter stemming. The scorer is created once and cached at
module level (it is cheap, but reuse keeps repeated calls allocation-free). The ``rouge_score``
import is lazy so the package imports fine without the optional ``[metrics]`` extra.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rouge_score.rouge_scorer import RougeScorer


@lru_cache(maxsize=1)
def _scorer() -> RougeScorer:
    try:
        from rouge_score.rouge_scorer import RougeScorer
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "ROUGE-L scoring requires the optional 'metrics' dependencies. "
            "Install them with: uv sync --extra metrics"
        ) from exc
    return RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, reference: str) -> float:
    """Longest-common-subsequence F-measure between prediction and reference, in [0, 1]."""
    return _scorer().score(reference, prediction)["rougeL"].fmeasure
