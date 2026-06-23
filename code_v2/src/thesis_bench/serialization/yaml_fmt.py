"""YAML serializer (structural). A human-readable JSON subset; indentation-based,
so noticeably fewer characters than JSON/XML.
"""

from __future__ import annotations

import yaml

from ..enums import SerializationFormat
from ._values import jsonable_row
from .snapshot import DatabaseSnapshot


class YamlSerializer:
    @property
    def fmt(self) -> SerializationFormat:
        return SerializationFormat.YAML

    def serialize(self, snapshot: DatabaseSnapshot) -> str:
        data = {
            table.name: [jsonable_row(row) for row in table.rows]
            for table in snapshot.tables
        }
        return yaml.safe_dump(data, indent=2, sort_keys=False, allow_unicode=True)
