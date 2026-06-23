"""Dataset-layer value types: a question and its executed ground-truth answer."""

from __future__ import annotations

from dataclasses import dataclass

from ..serialization.snapshot import Row


@dataclass(frozen=True, slots=True)
class QuestionItem:
    """A single natural-language question paired with its reference SQL.

    ``question_id`` is stable within a dataset and is used for resumability
    (skipping already-completed questions). ``gold_sql`` is the dataset-provided
    reference query, executed to build the ground truth.
    """

    question_id: str
    db_id: str
    question: str
    gold_sql: str


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The reference answer both pipelines are scored against.

    Built by executing ``QuestionItem.gold_sql`` against the *same* (possibly
    sampled) database the model sees — see ``db.materialize_sample`` — so the
    direct pipeline is never judged against rows it could not have seen.

    ``reference_text`` is the canonical stringification of ``rows`` (see
    ``eval.accuracy.rows_to_text``); both ROUGE-L and BERTScore compare against
    it, so text metrics on tabular output stay meaningful.
    """

    rows: tuple[Row, ...]
    reference_text: str
    error: str | None = None
