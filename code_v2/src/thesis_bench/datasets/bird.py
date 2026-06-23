"""BIRD dataset loader.

BIRD's questions file is a JSON list with ``db_id``, ``question``, and ``SQL``
(the gold query). Databases live at
``<db_root>/<db_id>/<db_id>.sqlite``. BIRD tables are large and unsanitized, so
they are sub-sampled to ``sample_rows`` rows per table (set in ``DatasetConfig``);
the deterministic sampling lives in ``db.sqlite.build_snapshot``.

BIRD entries also carry an ``evidence`` field (external knowledge). It is appended
to the question text so the model has the same hint the dataset intends, matching
ch-03's note that BIRD requires external knowledge.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..enums import DatasetKind
from ..errors import DatasetError
from .base import Dataset
from .types import QuestionItem


class BirdDataset(Dataset):
    kind = DatasetKind.BIRD

    def iter_questions(self) -> Iterator[QuestionItem]:
        path = self.config.questions_path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Could not load BIRD questions {path}: {exc}") from exc

        for index, entry in enumerate(raw):
            question = str(entry["question"])
            evidence = str(entry.get("evidence", "")).strip()
            if evidence:
                question = f"{question}\n\nExternal knowledge: {evidence}"

            yield QuestionItem(
                question_id=f"bird-{entry.get('question_id', index)}",
                db_id=str(entry["db_id"]),
                question=question,
                gold_sql=str(entry["SQL"]),
            )

    def db_path(self, db_id: str) -> Path:
        return self.config.db_root / db_id / f"{db_id}.sqlite"
