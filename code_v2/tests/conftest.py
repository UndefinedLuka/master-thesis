"""Shared fixtures: a tiny SQLite database with NULLs and a BLOB."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    """A 2-table SQLite DB exercising NULL, a BLOB and a few rows."""
    db = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, photo BLOB);
        CREATE TABLE pets (id INTEGER PRIMARY KEY, owner INTEGER, species TEXT);
        """
    )
    conn.execute(
        "INSERT INTO people (id, name, age, photo) VALUES (?,?,?,?)",
        (1, "Alice", 30, sqlite3.Binary(b"\x00\xff")),
    )
    conn.execute(
        "INSERT INTO people (id, name, age, photo) VALUES (?,?,?,?)",
        (2, "Bob", None, None),  # NULL age and NULL blob
    )
    conn.executemany(
        "INSERT INTO pets (id, owner, species) VALUES (?,?,?)",
        [(1, 1, "cat"), (2, 1, "dog"), (3, 2, "fish")],
    )
    conn.commit()
    conn.close()
    return db
