"""Markdown serialization (new in v2).

Emits one GitHub-Flavored-Markdown pipe table per database table, under an ``##`` heading.
Hand-rolled to avoid adding a dependency; included because Chapter 3 notes the widespread
presence of Markdown tables and LLMs' preference for them during text generation.
"""

from __future__ import annotations

from serialization.tabular import union_columns
from serialization.types import DatabaseContents, Row


def _cell(value: object) -> str:
    """Render a cell, escaping pipes/newlines so the table grid stays intact."""
    if value is None:
        return ""
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _table_to_markdown(name: str, rows: list[Row]) -> str:
    header = f"## {name}"
    if not rows:
        return f"{header}\n\n_(empty)_"

    columns = union_columns(rows)
    head_line = "| " + " | ".join(_cell(c) for c in columns) + " |"
    sep_line = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = ["| " + " | ".join(_cell(row.get(col)) for col in columns) + " |" for row in rows]
    return "\n".join([header, "", head_line, sep_line, *body_lines])


def to_markdown(contents: DatabaseContents) -> str:
    """One Markdown section + pipe table per table, separated by blank lines."""
    return "\n\n".join(_table_to_markdown(name, rows) for name, rows in contents.items())
