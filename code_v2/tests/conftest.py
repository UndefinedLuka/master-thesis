"""Shared pytest fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from thesis_bench.serialization.snapshot import DatabaseSnapshot, TableSnapshot


@pytest.fixture
def people_snapshot() -> DatabaseSnapshot:
    return DatabaseSnapshot(
        db_id="demo",
        tables=(
            TableSnapshot(
                name="people",
                columns=("id", "name", "age"),
                rows=(
                    {"id": 1, "name": "Alice", "age": 30},
                    {"id": 2, "name": "Bob", "age": 25},
                    {"id": 3, "name": "Carol", "age": 41},
                ),
            ),
        ),
        schema_sql="CREATE TABLE people (id INTEGER, name TEXT, age INTEGER);",
    )


@pytest.fixture
def big_db(tmp_path: Path) -> Path:
    """A SQLite file with 100 rows, for deterministic-sampling tests."""
    path = tmp_path / "big.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE nums (id INTEGER PRIMARY KEY, val TEXT)")
    conn.executemany(
        "INSERT INTO nums (id, val) VALUES (?, ?)",
        [(i, f"row-{i}") for i in range(100)],
    )
    conn.commit()
    conn.close()
    return path
