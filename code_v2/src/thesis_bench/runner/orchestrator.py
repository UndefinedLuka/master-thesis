"""The benchmark orchestrator.

Drives the experiment matrix produced by ``BenchmarkConfig.expand()``. For each
cell it runs every question through the same path — build snapshot, materialize
sample, build ground truth, execute the pipeline, score, record — so the two
pipelines share one code path and differ only in the pipeline object used.

Key behaviors:
- **Preflight**: before running an Ollama cell, verify the model tag exists
  locally so a missing model fails fast instead of triggering a multi-GB pull
  mid-run (which would also wreck timing).
- **Snapshot caching**: a database is captured once per cell and reused across
  all its questions (snapshots are deterministic and immutable).
- **Resumability**: questions already present in the cell's NDJSON are skipped.
"""

from __future__ import annotations

from collections.abc import Iterator

from tqdm import tqdm

from ..config import BenchmarkConfig, RunConfig
from ..datasets.base import Dataset
from ..datasets.factory import build_dataset
from ..datasets.types import QuestionItem
from ..db import sqlite
from ..enums import Backend, PipelineKind
from ..eval.metrics import score_record
from ..llm.base import LLMClient
from ..llm.factory import build_client
from ..llm.ollama_client import OllamaClient
from ..pipelines.base import Pipeline, PipelineContext
from ..pipelines.direct import DirectPipeline
from ..pipelines.text_to_sql import TextToSqlPipeline
from ..records import BenchmarkRecord, HostInfo
from ..serialization.snapshot import DatabaseSnapshot, Row
from .storage import ResultStore


class _ConnectionExecutor:
    """A ``SqlExecutor`` backed by an open materialized-sample connection."""

    def __init__(self, snapshot: DatabaseSnapshot) -> None:
        self._cm = sqlite.materialize_sample(snapshot)
        self._conn = self._cm.__enter__()

    def execute(self, sql: str) -> tuple[tuple[str, ...], tuple[Row, ...]]:
        return sqlite.execute_on_connection(self._conn, sql)

    def close(self) -> None:
        self._cm.__exit__(None, None, None)


def _make_pipeline(kind: PipelineKind) -> Pipeline:
    match kind:
        case PipelineKind.DIRECT:
            return DirectPipeline()
        case PipelineKind.TEXT_TO_SQL:
            return TextToSqlPipeline()


class Orchestrator:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._host = HostInfo.detect()

    def run(self) -> None:
        cells = self.config.expand()
        for cell in cells:
            self._run_cell(cell)

    def _preflight(self, client: LLMClient, cell: RunConfig) -> bool:
        """Return True if the cell can run; warn and skip otherwise."""
        if cell.model.backend is not Backend.OLLAMA:
            return True
        if not isinstance(client, OllamaClient):
            return True
        available = client.list_local_models()
        if cell.model.model_id not in available:
            tqdm.write(
                f"[skip] model {cell.model.model_id!r} not found in Ollama. "
                f"Run: ollama pull {cell.model.model_id}"
            )
            return False
        return True

    def _iter_questions(self, dataset: Dataset) -> Iterator[QuestionItem]:
        """Yield questions, optionally capped per database for quick runs."""
        limit = self.config.question_limit
        if limit is None:
            yield from dataset.iter_questions()
            return
        per_db: dict[str, int] = {}
        for item in dataset.iter_questions():
            count = per_db.get(item.db_id, 0)
            if count >= limit:
                continue
            per_db[item.db_id] = count + 1
            yield item

    def _run_cell(self, cell: RunConfig) -> None:
        dataset = build_dataset(cell.dataset)
        client = build_client(cell.model)
        if not self._preflight(client, cell):
            return

        store = ResultStore(self.config.results_dir, cell)
        done = store.completed_question_ids()
        pipeline = _make_pipeline(cell.pipeline)

        items = [it for it in self._iter_questions(dataset) if it.question_id not in done]
        snapshot_cache: dict[str, DatabaseSnapshot] = {}

        pbar = tqdm(items, desc=cell.cell_id, unit="q")
        for item in pbar:
            snapshot = snapshot_cache.get(item.db_id)
            if snapshot is None:
                snapshot = dataset.build_snapshot(item.db_id)
                snapshot_cache[item.db_id] = snapshot

            ground_truth = dataset.build_ground_truth(item, snapshot)

            executor = _ConnectionExecutor(snapshot)
            try:
                ctx = PipelineContext(
                    item=item,
                    snapshot=snapshot,
                    fmt=cell.fmt,
                    client=client,
                    options=cell.model.options,
                    executor=executor,
                    max_sql_retries=self.config.max_sql_retries,
                )
                result = pipeline.execute(ctx)
            finally:
                executor.close()

            scores = score_record(result, ground_truth, cell.pipeline)
            record = BenchmarkRecord.build(
                run=cell,
                item=item,
                reference_text=ground_truth.reference_text,
                result=result,
                scores=scores,
                host=self._host,
            )
            store.append(record)
