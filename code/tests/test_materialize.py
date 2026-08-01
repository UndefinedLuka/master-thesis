"""Tests for on-disk sampled-database materialization (BIRD path)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlite.materialize import materialize_sampled_db
from sqlite.sampling import load_full_database


def test_materialized_db_caps_rows_and_keeps_schema(tiny_db: Path, tmp_path: Path) -> None:
    dest = materialize_sampled_db(tiny_db, tmp_path / "out" / "tiny.sqlite", n_rows=2, seed=0)
    assert dest.exists()
    contents = load_full_database(dest)
    assert set(contents) == {"singer", "concert"}
    assert len(contents["singer"]) == 2  # capped
    assert len(contents["concert"]) == 2  # <= n_rows keeps all rows
    # sampled rows are a subset of the source rows
    source_names = {r["name"] for r in load_full_database(tiny_db)["singer"]}
    assert {r["name"] for r in contents["singer"]} <= source_names
    # schema (incl. the FK) copied verbatim
    conn = sqlite3.connect(dest)
    try:
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'concert'").fetchone()[0]
    finally:
        conn.close()
    assert "FOREIGN KEY" in ddl


def test_materialization_is_deterministic(tiny_db: Path, tmp_path: Path) -> None:
    a = materialize_sampled_db(tiny_db, tmp_path / "a.sqlite", n_rows=3, seed=7)
    b = materialize_sampled_db(tiny_db, tmp_path / "b.sqlite", n_rows=3, seed=7)
    assert load_full_database(a) == load_full_database(b)


def test_materialization_overwrites_cleanly(tiny_db: Path, tmp_path: Path) -> None:
    dest = tmp_path / "tiny.sqlite"
    materialize_sampled_db(tiny_db, dest, n_rows=2, seed=0)
    first = load_full_database(dest)
    materialize_sampled_db(tiny_db, dest, n_rows=2, seed=0)
    assert load_full_database(dest) == first
