"""XML serializer (structural). The most verbose format (explicit open/close
tags), common in legacy/enterprise systems.
"""

from __future__ import annotations

from dict2xml import dict2xml

from ..enums import SerializationFormat
from ..serialization.snapshot import Row
from ._values import jsonable
from .snapshot import DatabaseSnapshot


def _xml_row(row: Row) -> dict[str, str | int | float | bool]:
    # dict2xml renders ``None`` as the literal text "None"; map NULL to an empty
    # string instead so it isn't mistaken for data.
    out: dict[str, str | int | float | bool] = {}
    for key, val in row.items():
        coerced = jsonable(val)
        out[key] = "" if coerced is None else coerced
    return out


class XmlSerializer:
    @property
    def fmt(self) -> SerializationFormat:
        return SerializationFormat.XML

    def serialize(self, snapshot: DatabaseSnapshot) -> str:
        data = {
            table.name: [_xml_row(row) for row in table.rows]
            for table in snapshot.tables
        }
        return dict2xml(data, wrap="database", indent="  ")
