"""TOON (Token-Oriented Object Notation) serializer (token-optimized).

TOON separates schema from records and compresses redundant keys, minimizing
token overhead. Caveat (ch-03): it is new and under-represented in LLM training
corpora, which may impair inference — that trade-off is exactly what the
benchmark measures.
"""

from __future__ import annotations

from typing import Any

import toon

from ..enums import SerializationFormat
from ._values import jsonable_row
from .snapshot import DatabaseSnapshot


def _encode(data: dict[str, list[dict[str, Any]]]) -> str:
    # ``toon`` ships no type stubs; coerce the result to ``str`` at this boundary.
    return str(toon.encode(data))


class ToonSerializer:
    @property
    def fmt(self) -> SerializationFormat:
        return SerializationFormat.TOON

    def serialize(self, snapshot: DatabaseSnapshot) -> str:
        data = {
            table.name: [jsonable_row(row) for row in table.rows]
            for table in snapshot.tables
        }
        return _encode(data)
