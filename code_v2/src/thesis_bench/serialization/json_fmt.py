"""JSON serializer (structural). Verbose but ubiquitous in LLM training data."""

from __future__ import annotations

import json

from ..enums import SerializationFormat
from ._values import jsonable_row
from .snapshot import DatabaseSnapshot


class JsonSerializer:
    @property
    def fmt(self) -> SerializationFormat:
        return SerializationFormat.JSON

    def serialize(self, snapshot: DatabaseSnapshot) -> str:
        data = {
            table.name: [jsonable_row(row) for row in table.rows]
            for table in snapshot.tables
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
