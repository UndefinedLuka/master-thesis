"""CSV serializer (positional). Minimal overhead — headers written once, values
mapped by column position. Cannot represent nested data. Each table is emitted
as its own labelled CSV block so multiple tables coexist in one blob.
"""

from __future__ import annotations

import csv
import io

from ..enums import SerializationFormat
from ._values import cell_text
from .snapshot import DatabaseSnapshot, TableSnapshot


def _table_to_csv(table: TableSnapshot) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table.columns)
    for row in table.rows:
        writer.writerow([cell_text(row.get(col)) for col in table.columns])
    return buffer.getvalue()


class CsvSerializer:
    @property
    def fmt(self) -> SerializationFormat:
        return SerializationFormat.CSV

    def serialize(self, snapshot: DatabaseSnapshot) -> str:
        blocks = [
            f"# {table.name}\n{_table_to_csv(table)}" for table in snapshot.tables
        ]
        return "\n".join(blocks)
