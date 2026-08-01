"""Retrieval-accuracy metric primitives.

Pure where possible: ``rouge_l`` (rouge-score), ``bert_score_batch`` (bert-score; lazy
cached model), and ``execution_accuracy`` (Spider-style EX, dependency-free). The heavy
ROUGE/BERTScore deps live in the optional ``[metrics]`` extra and import lazily, so this
package still imports without them.

Scoring orchestration lives in :mod:`metrics.scoring` — imported directly rather
than re-exported here, because it depends on ``pipelines.types`` (which depends back on
``metrics.types``); keeping ``__init__`` to the leaf primitives avoids that import cycle.
"""

from metrics.bertscore import bert_score_batch
from metrics.execution import execution_accuracy, rows_to_canonical
from metrics.rouge import rouge_l
from metrics.types import AccuracyScores, ExecutionScore

__all__ = [
    "rouge_l",
    "bert_score_batch",
    "execution_accuracy",
    "rows_to_canonical",
    "AccuracyScores",
    "ExecutionScore",
]
