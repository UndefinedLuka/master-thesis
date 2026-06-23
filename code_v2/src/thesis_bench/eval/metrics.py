"""Answer-similarity and correctness scoring.

ROUGE-L (longest-common-subsequence overlap) is computed inline for every record
— it is cheap and pure-Python (``rouge-score``). BERTScore is heavy (torch +
transformers + a model download) and is therefore NOT computed here; it is run as
an optional post-pass over the results (see ``bertscore_pass``), so the benchmark
loop stays torch-free.

``score_record`` assembles the full ``EvalScores`` for one pipeline result given
its ground truth.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..datasets.types import GroundTruth
from ..enums import PipelineKind
from ..pipelines.types import PipelineResult
from .types import EvalScores


@lru_cache(maxsize=1)
def _scorer() -> Any:
    """Lazily build (and cache) the rouge-score scorer.

    Imported lazily so importing this module doesn't pull ``rouge_score`` until a
    score is actually requested. ``rouge_score`` ships no type stubs, so the
    scorer is typed ``Any`` here; ``rouge_l`` immediately coerces its output to a
    ``float``, keeping the untyped surface to this one function.
    """
    from rouge_score.rouge_scorer import RougeScorer

    return RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure in [0, 1].

    Two empty strings are treated as a perfect match (1.0); an empty prediction
    against a non-empty reference (or vice versa) scores 0.0.
    """
    pred = prediction.strip()
    ref = reference.strip()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    scores: Any = _scorer().score(ref, pred)
    return float(scores["rougeL"].fmeasure)


def score_record(
    result: PipelineResult,
    ground_truth: GroundTruth,
    pipeline: PipelineKind,
) -> EvalScores:
    """Compute the inline scores (ROUGE-L + execution accuracy) for one record.

    ``bertscore_f1`` is left ``None`` here; fill it via the post-pass if enabled.
    """
    rl = rouge_l(result.answer_text, ground_truth.reference_text)

    if pipeline is PipelineKind.TEXT_TO_SQL:
        execution_accuracy = bool(result.execution_ok)
        # With the canonical rows->text stringifier, identical reference and
        # answer text means the executed row set matches the gold row set.
        row_match = (
            execution_accuracy
            and result.answer_text == ground_truth.reference_text
        )
        return EvalScores(
            rouge_l=rl,
            bertscore_f1=None,
            execution_accuracy=execution_accuracy,
            row_set_match=row_match,
        )

    return EvalScores(rouge_l=rl, bertscore_f1=None)
