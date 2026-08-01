"""Load Spider dev examples.

Spider ``dev.json`` is a list of objects with ``question``, ``query`` (gold SQL) and
``db_id``. Spider has no external-knowledge evidence and no difficulty label.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets.types import DatasetExample


def load_spider_examples(dev_json: Path | str) -> list[DatasetExample]:
    with Path(dev_json).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    examples: list[DatasetExample] = []
    for index, item in enumerate(raw):
        db_id = item["db_id"]
        examples.append(
            DatasetExample(
                dataset="spider",
                db_id=db_id,
                question=item["question"],
                gold_sql=item["query"],
                evidence=None,
                difficulty=None,
                example_id=f"spider:{db_id}:{index}",
            )
        )
    return examples
