"""Build reference answers by executing each dataset's GOLD SQL query.

The reference "answer" for accuracy scoring is the gold query's result set,
serialized to text via :func:`metrics.rows_to_text`. Questions whose gold SQL
fails to execute are still emitted (with ``error`` set and an empty answer) so
they are never silently dropped; the runner flags them ``reference_unavailable``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from benchmark.config import DatasetSpec
from benchmark.datasets import load_questions
from benchmark.metrics import rows_to_text
from benchmark.sqlite_io import query_sqlite, to_records


@dataclass
class Reference:
    index: int
    db_id: str
    question: str
    gold_sql: str
    rows: list[dict] | None  # gold result set (None if the gold query errored)
    answer_text: str  # rows_to_text(rows); "" on error
    error: str | None


def build_references(spec: DatasetSpec, out_path: Path) -> Path:
    """Execute all gold queries for ``spec`` and write ``reference_<name>.json``."""
    questions = load_questions(spec)
    references: list[Reference] = []

    pbar = tqdm(questions, desc=f"Reference [{spec.name}]", unit="q")
    for q in pbar:
        db = spec.question_db_path(q.db_id)
        try:
            rows = to_records(query_sqlite(db, q.gold_sql))
            references.append(
                Reference(q.index, q.db_id, q.question, q.gold_sql, rows, rows_to_text(rows), None)
            )
        except Exception as e:  # noqa: BLE001 - record, never abort the whole build
            references.append(
                Reference(q.index, q.db_id, q.question, q.gold_sql, None, "", str(e))
            )
            pbar.set_postfix(last_error=q.db_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in references], f, ensure_ascii=False, indent=2)
    return out_path


def load_references(path: Path) -> dict[int, Reference]:
    """Load references keyed by question index."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["index"]: Reference(**r) for r in data}
