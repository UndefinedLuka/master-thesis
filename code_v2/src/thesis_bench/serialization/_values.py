"""Shared cell-value rendering helpers used across serializers.

Snapshot rows may contain ``bytes`` (BLOB columns) and ``None`` (NULL), neither
of which the structural encoders (JSON/YAML/TOON) accept directly. These helpers
normalize values consistently so every format renders the same logical data.
"""

from __future__ import annotations

from ..serialization.snapshot import CellValue, Row

# How NULL is rendered in positional/text formats. Structural formats keep ``None``.
NULL_TEXT = ""


def jsonable(value: CellValue) -> str | int | float | bool | None:
    """Coerce a cell value to a JSON/YAML/TOON-safe scalar.

    ``bytes`` are decoded leniently to text; everything else passes through.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def jsonable_row(row: Row) -> dict[str, str | int | float | bool | None]:
    return {key: jsonable(val) for key, val in row.items()}


def cell_text(value: CellValue) -> str:
    """Render a cell as plain text for positional formats (CSV/Markdown)."""
    if value is None:
        return NULL_TEXT
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
