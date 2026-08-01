"""Dataset-agnostic example type, shared by the Spider and BIRD loaders."""

from __future__ import annotations

from typing import TypedDict

from config.settings import DatasetName


class DatasetExample(TypedDict):
    """One natural-language question paired with its gold SQL."""

    dataset: DatasetName
    db_id: str
    question: str
    gold_sql: str
    evidence: str | None  # BIRD external-knowledge hint; None for Spider
    difficulty: str | None  # BIRD difficulty label if present; else None
    example_id: str  # stable id "<dataset>:<db_id>:<index>" for resumability
