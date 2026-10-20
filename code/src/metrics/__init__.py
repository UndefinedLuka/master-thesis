"""Retrieval-accuracy metric primitives.

Pure where possible: ``normalized_exact_match`` (Transformers' SQuAD metric), ``rouge_l``
(rouge-score), ``bert_score_batch`` (bert-score; lazy cached model), and binary
``execution_accuracy`` (dependency-free). Optional metric dependencies live in the
``[metrics]`` extra and import lazily, so this
package still imports without them.

Scoring orchestration lives in :mod:`metrics.scoring` — imported directly rather
than re-exported here, because it depends on ``pipelines.types`` (which depends back on
``metrics.types``); keeping ``__init__`` to the leaf primitives avoids that import cycle.
"""

from metrics.bertscore import bert_score_batch
from metrics.exact_match import normalized_exact_match
from metrics.execution import execution_accuracy
from metrics.rouge import rouge_l
from metrics.types import AccuracyScores, ExecutionScore

__all__ = [
    "normalized_exact_match",
    "rouge_l",
    "bert_score_batch",
    "execution_accuracy",
    "AccuracyScores",
    "ExecutionScore",
]
