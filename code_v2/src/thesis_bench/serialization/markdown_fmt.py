"""Markdown table serializer (positional).

LLMs prefer Markdown during generation, and pipe-delimited tables are a de-facto
standard. Like CSV, it is positional and cannot nest. Each table becomes a
heading plus a GitHub-flavored table. Pipe characters in values are escaped so
the table structure is preserved.
"""

from __future__ import annotations

from ..enums import SerializationFormat
from ._values import cell_text
from .snapshot import DatabaseSnapshot, TableSnapshot


def _escape(text: str) -> str:
    # Escape pipes and collapse newlines so a value stays within one table cell.
    return text.replace("|", "\\|").replace("\n", " ")


def _table_to_markdown(table: TableSnapshot) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    lines = [header, separator]
    for row in table.rows:
        cells = [_escape(cell_text(row.get(col))) for col in table.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class MarkdownSerializer:
    @property
    def fmt(self) -> SerializationFormat:
        return SerializationFormat.MARKDOWN

    def serialize(self, snapshot: DatabaseSnapshot) -> str:
        blocks = [
            f"## {table.name}\n\n{_table_to_markdown(table)}"
            for table in snapshot.tables
        ]
        return "\n\n".join(blocks)
