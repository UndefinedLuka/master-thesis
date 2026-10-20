"""Generate reference answers by executing each example's gold SQL against its database.

The resulting rows are the ground truth for BOTH pipelines (Chapter 3): Direct Context
Inference is scored against them as text, and Text-to-SQL's executed rows are compared to
them for text scoring. Failures of the gold query itself are captured (not raised) so one bad
example never aborts a long build.

For BIRD, ``cfg.db_root`` points at the MATERIALIZED sampled databases (built by
thesis-sample-dbs), so gold SQL runs over exactly the rows the model sees in the prompt and
the rows Text-to-SQL queries — the three consumers can no longer drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from tqdm import tqdm

from config.settings import DatasetConfig
from datasets.loader import resolve_db_path
from datasets.types import DatasetExample
from sqlite.connection import query_sqlite_safe


class ReferenceAnswer(TypedDict):
    example_id: str
    db_id: str
    question: str
    gold_sql: str
    rows: list[dict[str, Any]] | None  # gold result set; None if the gold query errored
    column_names: list[str] | None
    row_count: int | None
    error: str | None


def generate_reference_answer(example: DatasetExample, db_path: Path) -> ReferenceAnswer:
    """Execute one example's gold SQL and capture its result rows (or the error)."""
    result, error = query_sqlite_safe(db_path, example["gold_sql"])
    if result is None:
        return ReferenceAnswer(
            example_id=example["example_id"],
            db_id=example["db_id"],
            question=example["question"],
            gold_sql=example["gold_sql"],
            rows=None,
            column_names=None,
            row_count=None,
            error=error,
        )
    return ReferenceAnswer(
        example_id=example["example_id"],
        db_id=example["db_id"],
        question=example["question"],
        gold_sql=example["gold_sql"],
        rows=result.rows,
        column_names=result.columns,
        row_count=len(result.rows),
        error=None,
    )


def load_reference_file(path: Path) -> list[ReferenceAnswer]:
    """Load a reference NDJSON file (as written by :func:`write_reference_file`)."""
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    references: list[ReferenceAnswer] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                references.append(json.loads(line))
    return references


def write_reference_file(path: Path, references: list[ReferenceAnswer]) -> None:
    """Write references as NDJSON (overwrites), the format ``load_reference_file`` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for reference in references:
            handle.write(json.dumps(reference, ensure_ascii=False) + "\n")


def build_reference_set(
    examples: list[DatasetExample], cfg: DatasetConfig
) -> list[ReferenceAnswer]:
    """Generate reference answers for every example, with a progress bar.

    ``cfg.db_root`` must already point at the databases the experiment runs on — for BIRD
    that is the materialized sampled copies, so no in-memory re-sampling happens here.
    """
    references: list[ReferenceAnswer] = []
    pbar = tqdm(examples, desc=f"References ({cfg.name})", unit="q")
    failures = 0
    for example in pbar:
        reference = generate_reference_answer(example, resolve_db_path(example, cfg))
        if reference["error"] is not None:
            failures += 1
            pbar.set_postfix(failures=failures)
        references.append(reference)
    return references
