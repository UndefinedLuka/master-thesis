"""Tests for sqlite access, schema extraction, and BIRD sampling determinism."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlite.connection import (
    open_connection,
    query_connection,
    query_connection_safe,
    query_sqlite_safe,
)
from sqlite.sampling import build_sampled_connection, load_full_database, sample_database
from sqlite.schema import get_schema_ddl, get_table_names


def test_get_table_names(tiny_db: Path) -> None:
    assert set(get_table_names(tiny_db)) == {"singer", "concert"}


def test_get_schema_ddl(tiny_db: Path) -> None:
    ddl = get_schema_ddl(tiny_db)
    assert "CREATE TABLE singer" in ddl
    assert "CREATE TABLE concert" in ddl


def test_load_full_database(tiny_db: Path) -> None:
    contents = load_full_database(tiny_db)
    assert len(contents["singer"]) == 5
    assert len(contents["concert"]) == 2
    assert contents["singer"][0]["name"] == "Alice"


def test_sample_database_caps_rows(tiny_db: Path) -> None:
    contents = sample_database(tiny_db, n_rows=2, seed=0)
    assert len(contents["singer"]) == 2  # 5 rows -> capped at 2
    assert len(contents["concert"]) == 2  # 2 rows -> returned whole


def test_sample_database_is_deterministic(tiny_db: Path) -> None:
    a = sample_database(tiny_db, n_rows=3, seed=42)
    b = sample_database(tiny_db, n_rows=3, seed=42)
    assert a == b  # same seed -> identical sample (reused by both pipelines)


def test_sample_seed_changes_sample(tmp_path: Path) -> None:
    # A 1000-row table sampled down to 2: different seeds picking the same pair would be a
    # ~1-in-250k coincidence, so inequality is a deterministic expectation here.
    db_path = tmp_path / "big.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    conn.executemany("INSERT INTO t (id) VALUES (?)", [(i,) for i in range(1000)])
    conn.commit()
    conn.close()

    a = sample_database(db_path, n_rows=2, seed=1)
    b = sample_database(db_path, n_rows=2, seed=2)
    assert a["t"] != b["t"]


def test_query_sqlite_safe_reports_error(tiny_db: Path) -> None:
    df, error = query_sqlite_safe(tiny_db, "SELECT * FROM does_not_exist")
    assert df is None
    assert error is not None and "does_not_exist" in error


def test_query_sqlite_safe_success(tiny_db: Path) -> None:
    result, error = query_sqlite_safe(tiny_db, "SELECT name FROM singer WHERE id = 1")
    assert error is None
    assert result is not None
    assert result.rows == [{"name": "Alice"}]


def test_query_connection_reuses_open_connection(tiny_db: Path) -> None:
    conn = open_connection(tiny_db)
    try:
        result = query_connection(conn, "SELECT count(*) AS n FROM singer")
        assert result.rows == [{"n": 5}]
        assert result.columns == ["n"]
    finally:
        conn.close()


def test_query_connection_safe_reports_error(tiny_db: Path) -> None:
    conn = open_connection(tiny_db)
    try:
        result, error = query_connection_safe(conn, "SELECT * FROM nope")
        assert result is None
        assert error is not None and "nope" in error
    finally:
        conn.close()


def test_build_sampled_connection_runs_gold_sql(tiny_db: Path) -> None:
    # The in-memory sampled copy keeps the schema verbatim, so gold SQL runs unchanged
    # and sees exactly the sampled rows.
    conn = build_sampled_connection(tiny_db, n_rows=2, seed=0)
    try:
        result = query_connection(conn, "SELECT count(*) AS n FROM singer")
        assert result.rows == [{"n": 2}]
        names = query_connection(conn, "SELECT name FROM singer ORDER BY id").rows
        full = load_full_database(tiny_db)["singer"]
        assert all(any(r["name"] == row["name"] for row in full) for r in names)
    finally:
        conn.close()


def test_sample_without_rowid_table_falls_back(tmp_path: Path) -> None:
    db_path = tmp_path / "worid.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE kv (k TEXT PRIMARY KEY, v INTEGER) WITHOUT ROWID;
        INSERT INTO kv (k, v) VALUES ('a', 1), ('b', 2), ('c', 3), ('d', 4);
        """
    )
    conn.commit()
    conn.close()

    contents = sample_database(db_path, n_rows=2, seed=0)
    assert len(contents["kv"]) == 2  # sampled via the full-load fallback
    assert sample_database(db_path, n_rows=2, seed=0) == contents  # still deterministic
