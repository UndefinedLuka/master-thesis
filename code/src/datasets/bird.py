"""Load BIRD dev examples.

BIRD ``dev.json`` is a list of objects with ``question``, ``SQL`` (gold SQL), ``db_id``,
``evidence`` (external-knowledge hint), and often ``difficulty``. Key names have varied
across releases, so we read a few aliases defensively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets.types import DatasetExample


def _first(item: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return default


def load_bird_examples(dev_json: Path | str) -> list[DatasetExample]:
    with Path(dev_json).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    examples: list[DatasetExample] = []
    for index, item in enumerate(raw):
        db_id = item["db_id"]
        gold_sql = _first(item, ("SQL", "query", "sql"), default="")
        examples.append(
            DatasetExample(
                dataset="bird",
                db_id=db_id,
                question=item["question"],
                gold_sql=gold_sql,
                evidence=_first(item, ("evidence",)),
                difficulty=_first(item, ("difficulty",)),
                example_id=f"bird:{db_id}:{index}",
            )
        )
    return examples
