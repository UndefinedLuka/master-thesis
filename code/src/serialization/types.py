"""Shared serialization types.

``DatabaseContents`` is the single plain-Python representation every serializer consumes,
decoupling the format encoders from sqlite/polars. Per Chapter 3 ("Structural Integrity"),
a database is a flat mapping of table name -> list of row dicts (no deep nesting), so all
six formats — including the positional ones (CSV, Markdown) — can represent it faithfully.
"""

from __future__ import annotations

from typing import Any, Literal

SerializationFormat = Literal["json", "xml", "yaml", "csv", "markdown", "toon"]

#: All formats, in the order presented in Chapter 3.
FORMATS: tuple[SerializationFormat, ...] = (
    "json",
    "xml",
    "yaml",
    "csv",
    "markdown",
    "toon",
)

#: table name -> list of rows, each row a column-name -> value mapping.
Row = dict[str, Any]
DatabaseContents = dict[str, list[Row]]

#: File extension per format (Markdown -> .md; the rest match their name).
FORMAT_EXTENSIONS: dict[SerializationFormat, str] = {
    "json": "json",
    "xml": "xml",
    "yaml": "yaml",
    "csv": "csv",
    "markdown": "md",
    "toon": "toon",
}
