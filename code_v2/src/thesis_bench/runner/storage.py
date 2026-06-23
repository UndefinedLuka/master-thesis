"""Resumable NDJSON result storage — one file per matrix cell.

Each ``RunConfig`` cell writes to ``<results_dir>/<cell_id>.ndjson``. Records are
appended one JSON object per line. Resumability works by reading back the set of
``question_id``s already present, so an interrupted run skips completed questions
on restart. One file per cell (rather than one giant file) keeps reads cheap and
lets cells run independently / in parallel.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import RunConfig
from ..records import BenchmarkRecord


class ResultStore:
    """Append-only NDJSON store for a single matrix cell."""

    def __init__(self, results_dir: Path, run: RunConfig) -> None:
        self._dir = results_dir
        self._path = results_dir / f"{run.cell_id}.ndjson"

    @property
    def path(self) -> Path:
        return self._path

    def completed_question_ids(self) -> set[str]:
        """IDs of questions already recorded in this cell's file (for resume)."""
        if not self._path.exists():
            return set()
        done: set[str] = set()
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = obj.get("question_id")
                if isinstance(qid, str):
                    done.add(qid)
        return done

    def append(self, record: BenchmarkRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
