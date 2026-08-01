"""Low-level SQLite querying via the stdlib sqlite3 module.

``query_sqlite`` carries v1's robust settings (decode-with-replace text factory so dirty
BIRD bytes don't crash). ``query_sqlite_safe`` is the non-raising variant the Text-to-SQL
error-correction loop and the execution metric build on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, NamedTuple


class QueryResult(NamedTuple):
    """Rows (as column-name -> value dicts) plus the column names in SELECT order."""

    rows: list[dict[str, Any]]
    columns: list[str]


def open_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open ``db_path`` with the project's robust text handling (decode-with-replace)."""
    conn = sqlite3.connect(str(db_path))
    conn.text_factory = lambda b: b.decode(errors="replace")  # type: ignore[assignment]
    return conn


def query_connection(conn: sqlite3.Connection, query: str) -> QueryResult:
    """Execute ``query`` against an already-open connection (caller owns lifetime)."""
    cur = conn.execute(query)
    columns = [d[0] for d in (cur.description or [])]
    rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    return QueryResult(rows=rows, columns=columns)


def query_connection_safe(
    conn: sqlite3.Connection, query: str
) -> tuple[QueryResult | None, str | None]:
    """Non-raising variant of :func:`query_connection`."""
    try:
        return query_connection(conn, query), None
    except Exception as exc:
        return None, str(exc)


def query_sqlite(db_path: Path | str, query: str) -> QueryResult:
    """Execute ``query`` against the SQLite file and return a :class:`QueryResult`.

    Raises on SQL/connection errors — use :func:`query_sqlite_safe` when failures are
    expected (untrusted, model-generated SQL).
    """
    conn = open_connection(db_path)
    try:
        return query_connection(conn, query)
    finally:
        conn.close()


def query_sqlite_safe(db_path: Path | str, query: str) -> tuple[QueryResult | None, str | None]:
    """Execute ``query``, returning ``(result, None)`` on success or ``(None, error)``."""
    try:
        return query_sqlite(db_path, query), None
    except Exception as exc:
        return None, str(exc)
