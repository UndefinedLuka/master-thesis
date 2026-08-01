"""Loading database contents for serialization.

* Spider: every row of every table (no sub-sampling) — :func:`load_full_database`.
* BIRD: ``n`` random rows per table, deterministic given ``seed`` — :func:`sample_database`.
  The same sampled rows are reused by both pipelines (Chapter 3), which determinism
  guarantees: the sample depends only on (table rowids, n, seed).

Sampling is memory-light: only the integer ``rowid`` column is loaded to choose the sample,
then exactly the chosen rows are fetched — we never load a large BIRD table in full just to
keep ``n`` rows. Tables declared ``WITHOUT ROWID`` (rare) fall back to loading + sampling.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

from serialization.types import DatabaseContents, Row
from sqlite.connection import open_connection, query_sqlite, query_sqlite_safe
from sqlite.schema import get_table_names


def load_full_database(db_path: Path | str) -> DatabaseContents:
    """Load all rows of all tables (Spider path)."""
    contents: DatabaseContents = {}
    for name in get_table_names(db_path):
        contents[name] = query_sqlite(db_path, f'SELECT * FROM "{name}"').rows
    return contents


def sample_database(db_path: Path | str, n_rows: int, *, seed: int) -> DatabaseContents:
    """Load up to ``n_rows`` deterministically-random rows per table (BIRD path)."""
    contents: DatabaseContents = {}
    for index, name in enumerate(get_table_names(db_path)):
        contents[name] = _sample_table(db_path, name, n_rows, seed=seed + index)
    return contents


def _sample_table(db_path: Path | str, table: str, n_rows: int, *, seed: int) -> list[Row]:
    result, error = query_sqlite_safe(db_path, f'SELECT rowid AS __rid FROM "{table}"')
    if error is not None or result is None:
        # WITHOUT ROWID table (or no rowid): load fully, then sample in Python.
        return _sample_full_table(db_path, table, n_rows, seed=seed)

    rowids: list[int] = [r["__rid"] for r in result.rows]
    if not rowids:
        return []
    if len(rowids) <= n_rows:
        chosen = rowids
    else:
        chosen = random.Random(seed).sample(rowids, n_rows)

    placeholders = ", ".join(str(int(rid)) for rid in chosen)
    return query_sqlite(db_path, f'SELECT * FROM "{table}" WHERE rowid IN ({placeholders})').rows


def _sample_full_table(db_path: Path | str, table: str, n_rows: int, *, seed: int) -> list[Row]:
    rows = query_sqlite(db_path, f'SELECT * FROM "{table}"').rows
    if len(rows) <= n_rows:
        return rows
    return random.Random(seed).sample(rows, n_rows)


def build_sampled_connection(db_path: Path | str, n_rows: int, *, seed: int) -> sqlite3.Connection:
    """Build an in-memory SQLite database containing only the sampled rows.

    Returns an open connection the caller must close. The schema is copied verbatim from the
    source so gold SQL (which references column names and types) works unchanged. Used by
    reference-answer generation for BIRD so the gold query runs over the same subset of rows
    that will be presented to the model.
    """
    contents = sample_database(db_path, n_rows, seed=seed)

    # Fetch DDL statements directly from sqlite_master rather than going through
    # get_schema_ddl (which joins them into one string) — executing each statement
    # individually avoids problems with semicolons inside CHECK/DEFAULT expressions.
    src_conn = open_connection(db_path)
    try:
        cur = src_conn.execute(
            "SELECT sql FROM sqlite_master"
            " WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            " ORDER BY rowid;"
        )
        ddl_statements = [row[0] for row in cur.fetchall() if row[0]]
    finally:
        src_conn.close()

    mem_conn = sqlite3.connect(":memory:")
    try:
        for stmt in ddl_statements:
            mem_conn.execute(stmt)
        for table, rows in contents.items():
            if not rows:
                continue
            columns = list(rows[0].keys())
            placeholders = ", ".join("?" * len(columns))
            col_list = ", ".join(f'"{c}"' for c in columns)
            mem_conn.executemany(
                f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})',
                [tuple(row[c] for c in columns) for row in rows],
            )
        mem_conn.commit()
    except Exception:
        mem_conn.close()  # don't leak the connection when schema copy / insert fails
        raise
    return mem_conn
