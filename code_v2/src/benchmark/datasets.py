"""Question loading for Spider and BIRD into a uniform, indexed representation.

The flat indexed list (questions in file order) is what makes resume trivial:
line N of an NDJSON shard corresponds to ``Question.index == N``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from benchmark.config import DatasetSpec


@dataclass(frozen=True)
class Question:
    index: int  # stable position within the dataset (file order)
    db_id: str
    question: str
    gold_sql: str
    evidence: str | None = None  # BIRD external-knowledge hint; None for Spider


def _gold_sql(entry: dict) -> str:
    # Spider uses "query"; BIRD uses "SQL".
    sql = entry.get("query") or entry.get("SQL") or ""
    return sql.strip()


def load_questions(spec: DatasetSpec) -> list[Question]:
    """Load all questions for ``spec`` in file order."""
    with open(spec.questions_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    questions: list[Question] = []
    for i, entry in enumerate(entries):
        evidence = None
        if spec.has_evidence:
            ev = entry.get("evidence")
            evidence = ev.strip() if isinstance(ev, str) and ev.strip() else None
        questions.append(
            Question(
                index=i,
                db_id=entry["db_id"],
                question=entry["question"].strip(),
                gold_sql=_gold_sql(entry),
                evidence=evidence,
            )
        )
    return questions
