"""CSV serialization.

v1 wrote one CSV file per table; v2 returns a single string with each table as a labeled
block (``# table: <name>`` header + CSV) so every format shares the same in-memory ``str``
contract and the prompt builder treats all six formats uniformly.
"""

from __future__ import annotations

import csv
import io

from serialization.tabular import union_columns
from serialization.types import DatabaseContents, Row


def _cell(value: object) -> object:
    """CSV cannot nest; render the rare dict/list cell as a string. ``None`` -> empty."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return value


def _table_to_csv(rows: list[Row]) -> str:
    if not rows:
        return ""
    fieldnames = union_columns(rows)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(row.get(key)) for key in fieldnames})
    return buffer.getvalue().rstrip("\r\n")


def to_csv(contents: DatabaseContents) -> str:
    """Concatenate per-table CSV blocks, each prefixed by a ``# table:`` label."""
    blocks: list[str] = []
    for table_name, rows in contents.items():
        body = _table_to_csv(rows)
        blocks.append(f"# table: {table_name}\n{body}" if body else f"# table: {table_name}\n")
    return "\n\n".join(blocks)
