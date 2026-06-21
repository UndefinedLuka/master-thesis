"""Result row schema and resumable NDJSON I/O.

One :class:`ResultRow` is one NDJSON line. Resume is keyed by ``question_index``
(robust to partial writes and out-of-order shards) rather than a raw line count.
``bertscore`` is left ``None`` during a run and filled in batched during
aggregation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResultRow:
    # Identity
    model: str
    dataset: str
    pipeline: str  # "direct" | "sql"
    format: str | None  # serialization format (direct) or None (sql)
    question_index: int
    db_id: str
    question: str

    # Text compared by the accuracy metrics
    output_text: str = ""
    reference_text: str = ""
    reference_unavailable: bool = False

    # Operational metrics
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    total_generation_s: float = 0.0

    # Accuracy
    rouge_l: float = 0.0
    bertscore: float | None = None  # filled in aggregation

    # Text-to-SQL specific
    final_sql: str | None = None
    retry_count: int = 0
    definitive_failure: bool = False
    execution_accuracy: bool = False  # query executed without error
    result_set_match: bool = False  # rows match the gold result set
    exec_time_s: float = 0.0

    error: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def append_row(path: Path, row: ResultRow) -> None:
    """Append one row as a flushed NDJSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(row.to_json_line() + "\n")
        f.flush()


def already_done_indices(path: Path) -> set[int]:
    """Return the set of ``question_index`` values already present in the shard."""
    if not path.exists():
        return set()
    done: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["question_index"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a crash
    return done
