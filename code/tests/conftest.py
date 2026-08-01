"""Shared fixtures: a tiny on-disk SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    """A 2-table SQLite database with a foreign key and a handful of rows."""
    db_path = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        CREATE TABLE concert (
            id INTEGER PRIMARY KEY, singer_id INTEGER, city TEXT,
            FOREIGN KEY (singer_id) REFERENCES singer(id)
        );
        INSERT INTO singer (id, name, age) VALUES
            (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Cara', 41),
            (4, 'Dan', 19), (5, 'Eve', 33);
        INSERT INTO concert (id, singer_id, city) VALUES
            (10, 1, 'Zurich'), (11, 2, 'Bern');
        """
    )
    conn.commit()
    conn.close()
    return db_path
