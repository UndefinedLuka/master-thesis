"""Pipeline ABC and the shared execution context.

Both pipelines implement ``execute(ctx) -> PipelineResult`` so the orchestrator's
path is identical regardless of pipeline: build a snapshot, materialize the
sample, build the ground truth, call ``pipeline.execute``, score, record. Only
the middle step differs, which is exactly what the template-method ABC captures.

We use an ABC here (not a Protocol) because the two pipelines are an internal,
closed hierarchy that shares behavior; the Protocol seams are reserved for the
genuinely swappable boundaries (LLM backend, serializer, SQL executor).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config import GenerationOptions
from ..datasets.types import QuestionItem
from ..enums import PipelineKind, SerializationFormat
from ..llm.base import LLMClient
from ..serialization.snapshot import DatabaseSnapshot, Row
from .types import PipelineResult


@runtime_checkable
class SqlExecutor(Protocol):
    """Executes SQL against the materialized sample.

    Implementations raise ``SqlExecutionError`` on failure (so the retry loop can
    feed the message back) and return ``(columns, rows)`` on success. This thin
    seam keeps the Text-to-SQL pipeline independent of sqlite3 and easy to fake
    in tests.
    """

    def execute(self, sql: str) -> tuple[tuple[str, ...], tuple[Row, ...]]: ...


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Everything a pipeline needs for one question.

    ``executor`` is only used by the Text-to-SQL pipeline; the direct pipeline
    ignores it. Bundling them keeps ``execute`` to a single, uniform parameter.
    """

    item: QuestionItem
    snapshot: DatabaseSnapshot
    fmt: SerializationFormat
    client: LLMClient
    options: GenerationOptions
    executor: SqlExecutor
    max_sql_retries: int


class Pipeline(ABC):
    """Base class for the two retrieval pipelines."""

    #: Set by each subclass; identifies the pipeline in records.
    kind: PipelineKind

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineResult:
        """Run the pipeline for one question and return a uniform result."""
        ...
