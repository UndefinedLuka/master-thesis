"""SQLite access and the single value-coercion point for serialization.

``to_records`` is the linchpin: every serializer and the reference builder run
their rows through it so that NULLs, BLOBs and non-finite floats are handled
identically and can never crash a particular format (dict2xml and toon, for
instance, choke on raw ``bytes``; ``json`` chokes on ``NaN``/``inf``).
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl


def query_sqlite(path: str | Path, query: str) -> pl.DataFrame:
    """Run ``query`` against the SQLite file at ``path`` and return a DataFrame.

    ``text_factory`` decodes bytes with ``errors="replace"`` so mojibake in the
    (real-world, messy) BIRD databases does not raise.
    """
    conn = sqlite3.connect(str(path))
    conn.text_factory = lambda b: b.decode(errors="replace")  # type: ignore[assignment]
    try:
        return pl.read_database(query=query, connection=conn, infer_schema_length=None)
    finally:
        conn.close()


def coerce_value(value: Any) -> Any:
    """Coerce a single cell to a JSON/YAML/XML/TOON-safe scalar.

    - ``None`` stays ``None`` (renders as null / empty cell).
    - ``bytes`` -> ``"<BLOB:n bytes>"`` placeholder (binary content is never
      meaningful as text and breaks several serializers).
    - non-finite floats (``NaN``/``inf``) -> ``None`` (invalid in JSON).
    - ``Decimal`` -> ``float`` when finite, else ``str``.
    - ``date``/``datetime`` -> ISO string.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<BLOB:{len(bytes(value))} bytes>"
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        f = float(value)
        return f if math.isfinite(f) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def to_records(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of coerced row dicts."""
    return [{k: coerce_value(v) for k, v in row.items()} for row in df.to_dicts()]
