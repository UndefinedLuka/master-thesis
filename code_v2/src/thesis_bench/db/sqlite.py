"""SQLite access: query, schema extraction, deterministic sampling, and
in-memory materialization of a snapshot.

Uses the stdlib ``sqlite3`` module directly (no Polars) for full typing control
and zero heavy dependencies. Two correctness points carried over / fixed from v1:

1. ``text_factory`` decodes bytes with ``errors="replace"`` so dirty real-world
   data (BIRD) doesn't crash a run on a bad byte.
2. Sampling is **deterministic**: a stable per-table seed (``zlib.crc32`` — not
   Python's per-process-salted ``hash``) drives ``random.Random``. We never use
   ``ORDER BY RANDOM()``, which is not reproducible.
"""

from __future__ import annotations

import random
import sqlite3
import zlib
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from ..errors import SqlExecutionError
from ..serialization.snapshot import (
    CellValue,
    DatabaseSnapshot,
    Row,
    TableSnapshot,
)

# Identifier used to namespace the sampling seed so changing the salt globally
# reshuffles every sample reproducibly if ever needed.
_SAMPLE_SALT = "thesis_bench.sample.v1"


def _decode_lenient(raw: bytes) -> str:
    """Text factory: BIRD contains non-UTF-8 byte sequences; never crash on them."""
    return raw.decode("utf-8", errors="replace")


@contextmanager
def open_connection(path: Path) -> Generator[sqlite3.Connection]:
    """Open a connection with robust text decoding."""
    conn = sqlite3.connect(str(path))
    conn.text_factory = _decode_lenient
    try:
        yield conn
    finally:
        conn.close()


def _execute(conn: sqlite3.Connection, query: str) -> tuple[tuple[str, ...], list[Row]]:
    """Run ``query`` and return (column_names, rows-as-dicts).

    Raises ``SqlExecutionError`` (preserving the sqlite message) on failure so
    the Text-to-SQL retry loop can feed the message back to the model.
    """
    try:
        cur = conn.execute(query)
    except sqlite3.Error as exc:
        raise SqlExecutionError(str(exc)) from exc

    description = cur.description or ()
    columns = tuple(col[0] for col in description)
    rows: list[Row] = [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]
    return columns, rows


def query(path: Path, sql: str) -> tuple[tuple[str, ...], tuple[Row, ...]]:
    """Execute ``sql`` against the database file and return columns + rows."""
    with open_connection(path) as conn:
        columns, rows = _execute(conn, sql)
    return columns, tuple(rows)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    """Names of real (non-internal) tables, in database order."""
    _, rows = _execute(
        conn,
        "SELECT name FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
        "ORDER BY rowid;",
    )
    return [str(r["name"]) for r in rows]


def dump_schema(conn: sqlite3.Connection) -> str:
    """The concatenated ``CREATE TABLE`` DDL — fed to the Text-to-SQL prompt."""
    _, rows = _execute(
        conn,
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
        "ORDER BY rowid;",
    )
    return "\n\n".join(str(r["sql"]) for r in rows)


def _table_seed(base_seed: int, db_id: str, table: str) -> int:
    """Stable, process-independent seed for one (database, table) pair."""
    key = f"{_SAMPLE_SALT}|{base_seed}|{db_id}|{table}".encode()
    return zlib.crc32(key)


def _read_table(
    conn: sqlite3.Connection,
    db_id: str,
    table: str,
    sample_rows: int | None,
    base_seed: int,
) -> TableSnapshot:
    """Read a full table, or a deterministic ``sample_rows``-row sample of it."""
    columns, rows = _execute(conn, f'SELECT * FROM "{table}"')

    if sample_rows is not None and len(rows) > sample_rows:
        rng = random.Random(_table_seed(base_seed, db_id, table))
        # sorted indices keep the original row order within the sample, which
        # makes positional formats (CSV/Markdown) deterministic across runs.
        chosen = sorted(rng.sample(range(len(rows)), sample_rows))
        rows = [rows[i] for i in chosen]

    return TableSnapshot(name=table, columns=columns, rows=tuple(rows))


def build_snapshot(
    path: Path,
    db_id: str,
    *,
    sample_rows: int | None = None,
    sample_seed: int = 42,
) -> DatabaseSnapshot:
    """Capture a database (or a deterministic sample of it) into memory.

    ``sample_rows=None`` captures full tables (Spider). An int caps each table to
    ``n`` rows (BIRD). The same snapshot is then serialized into every format and
    materialized for query execution, guaranteeing one consistent sample.
    """
    with open_connection(path) as conn:
        schema_sql = dump_schema(conn)
        tables = tuple(
            _read_table(conn, db_id, name, sample_rows, sample_seed)
            for name in list_tables(conn)
        )
    return DatabaseSnapshot(
        db_id=db_id,
        tables=tables,
        schema_sql=schema_sql,
        sample_rows=sample_rows,
        sample_seed=sample_seed if sample_rows is not None else None,
    )


def _sqlite_type(values: list[CellValue]) -> str:
    """Best-effort column affinity from observed values (INTEGER/REAL/BLOB/TEXT)."""
    seen = [v for v in values if v is not None]
    if not seen:
        return "TEXT"
    if all(isinstance(v, bool) or isinstance(v, int) for v in seen):
        return "INTEGER"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seen):
        return "REAL"
    if all(isinstance(v, bytes) for v in seen):
        return "BLOB"
    return "TEXT"


@contextmanager
def materialize_sample(snapshot: DatabaseSnapshot) -> Generator[sqlite3.Connection]:
    """Rebuild ``snapshot`` as an in-memory SQLite database.

    This is the heart of ch-03's fairness requirement for BIRD: both the
    reference gold query and the model's generated SQL execute against the exact
    rows the model was shown, not the full original database. For Spider (full
    snapshot) the materialized DB is simply a faithful copy of the relevant data.

    Note: we recreate tables from observed column types rather than the original
    DDL, because foreign-key constraints would reject a partial sample. Queries
    only need the columns and rows, which this preserves.
    """
    conn = sqlite3.connect(":memory:")
    conn.text_factory = _decode_lenient
    try:
        for table in snapshot.tables:
            col_types = {
                col: _sqlite_type([row.get(col) for row in table.rows])
                for col in table.columns
            }
            col_defs = ", ".join(f'"{c}" {col_types[c]}' for c in table.columns)
            conn.execute(f'CREATE TABLE "{table.name}" ({col_defs})')

            if table.rows:
                placeholders = ", ".join("?" for _ in table.columns)
                col_list = ", ".join(f'"{c}"' for c in table.columns)
                insert = (
                    f'INSERT INTO "{table.name}" ({col_list}) '
                    f"VALUES ({placeholders})"
                )
                conn.executemany(
                    insert,
                    [tuple(row.get(c) for c in table.columns) for row in table.rows],
                )
        conn.commit()
        yield conn
    finally:
        conn.close()


def execute_on_connection(
    conn: sqlite3.Connection, sql: str
) -> tuple[tuple[str, ...], tuple[Row, ...]]:
    """Run ``sql`` on an already-open connection (e.g. a materialized sample)."""
    columns, rows = _execute(conn, sql)
    return columns, tuple(rows)
