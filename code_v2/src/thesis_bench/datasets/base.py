"""Dataset ABC and shared logic for Spider/BIRD.

A dataset knows how to (a) iterate its questions and (b) locate each database's
SQLite file. The base class provides the shared machinery that both concrete
datasets reuse: building a snapshot (with the dataset's sampling policy) and
building the ground truth by executing the gold query against the *materialized
sample* (not the original file), which is the ch-03 fairness requirement.

Subclasses only differ in how they parse their questions file and where their DB
files live, so the snapshot/ground-truth path stays identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from ..config import DatasetConfig
from ..db import sqlite
from ..enums import DatasetKind
from ..errors import SqlExecutionError
from ..eval.accuracy import rows_to_text
from ..serialization.snapshot import DatabaseSnapshot
from .types import GroundTruth, QuestionItem


class Dataset(ABC):
    """Base for a Text-to-SQL dataset (Spider, BIRD)."""

    kind: DatasetKind

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    @abstractmethod
    def iter_questions(self) -> Iterator[QuestionItem]:
        """Yield each question with its gold SQL, in file order."""
        ...

    @abstractmethod
    def db_path(self, db_id: str) -> Path:
        """Filesystem path to the SQLite database for ``db_id``."""
        ...

    def build_snapshot(self, db_id: str) -> DatabaseSnapshot:
        """Capture the database (full for Spider, sampled for BIRD)."""
        return sqlite.build_snapshot(
            self.db_path(db_id),
            db_id,
            sample_rows=self.config.sample_rows,
            sample_seed=self.config.sample_seed,
        )

    def build_ground_truth(
        self, item: QuestionItem, snapshot: DatabaseSnapshot
    ) -> GroundTruth:
        """Execute the gold SQL against the materialized sample to get the
        reference answer both pipelines are scored against.
        """
        try:
            with sqlite.materialize_sample(snapshot) as conn:
                columns, rows = sqlite.execute_on_connection(conn, item.gold_sql)
        except SqlExecutionError as exc:
            return GroundTruth(rows=(), reference_text="", error=exc.db_message)

        return GroundTruth(
            rows=rows,
            reference_text=rows_to_text(columns, rows),
            error=None,
        )
