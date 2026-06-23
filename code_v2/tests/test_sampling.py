"""Deterministic sampling and in-memory materialization."""

from __future__ import annotations

from pathlib import Path

from thesis_bench.db import sqlite


def test_sampling_is_deterministic(big_db: Path) -> None:
    snap1 = sqlite.build_snapshot(big_db, "big", sample_rows=10, sample_seed=42)
    snap2 = sqlite.build_snapshot(big_db, "big", sample_rows=10, sample_seed=42)
    ids1 = [r["id"] for r in snap1.tables[0].rows]
    ids2 = [r["id"] for r in snap2.tables[0].rows]
    assert ids1 == ids2
    assert len(ids1) == 10


def test_different_seed_changes_sample(big_db: Path) -> None:
    a = sqlite.build_snapshot(big_db, "big", sample_rows=10, sample_seed=1)
    b = sqlite.build_snapshot(big_db, "big", sample_rows=10, sample_seed=2)
    assert [r["id"] for r in a.tables[0].rows] != [r["id"] for r in b.tables[0].rows]


def test_no_sampling_returns_full_table(big_db: Path) -> None:
    snap = sqlite.build_snapshot(big_db, "big", sample_rows=None)
    assert snap.tables[0].row_count == 100
    assert snap.sample_rows is None


def test_materialize_sample_executes_queries(big_db: Path) -> None:
    snap = sqlite.build_snapshot(big_db, "big", sample_rows=5, sample_seed=7)
    with sqlite.materialize_sample(snap) as conn:
        cols, rows = sqlite.execute_on_connection(conn, "SELECT COUNT(*) AS n FROM nums")
    assert cols == ("n",)
    # The materialized DB contains exactly the sampled rows, not all 100.
    assert rows[0]["n"] == 5
