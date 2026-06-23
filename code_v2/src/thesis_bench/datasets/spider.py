"""Spider dataset loader.

Spider's questions file is a JSON list of objects with ``db_id``, ``question``,
and ``query`` (the gold SQL). Databases live at
``<db_root>/<db_id>/<db_id>.sqlite``. Spider tables are used in full
(``sample_rows`` is ``None``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..enums import DatasetKind
from ..errors import DatasetError
from .base import Dataset
from .types import QuestionItem


class SpiderDataset(Dataset):
    kind = DatasetKind.SPIDER

    def iter_questions(self) -> Iterator[QuestionItem]:
        path = self.config.questions_path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Could not load Spider questions {path}: {exc}") from exc

        for index, entry in enumerate(raw):
            yield QuestionItem(
                question_id=f"spider-{index}",
                db_id=str(entry["db_id"]),
                question=str(entry["question"]),
                gold_sql=str(entry["query"]),
            )

    def db_path(self, db_id: str) -> Path:
        return self.config.db_root / db_id / f"{db_id}.sqlite"
