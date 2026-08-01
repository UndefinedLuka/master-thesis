"""Materialize a sampled database to disk (BIRD path).

Chapter 3 requires that BOTH pipelines see the same sub-sampled BIRD data: Direct Context
Inference serializes the sampled rows into the prompt, and Text-to-SQL executes model SQL
against a database containing exactly those rows. Materializing the sample once to disk makes
that consistency structural — references, serialization, and query execution all read the same
file — and keeps Text-to-SQL query timing realistic (on-disk, not ``:memory:``).

Determinism: the sample depends only on (table rowids, n_rows, seed), so re-materializing
with the same seed reproduces the same database byte-for-byte at the row level.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlite.sampling import build_sampled_connection


def materialize_sampled_db(src: Path | str, dest: Path | str, n_rows: int, *, seed: int) -> Path:
    """Write a sampled copy of ``src`` to ``dest`` and return ``dest``.

    Builds the sampled database in memory (schema copied verbatim, ``n_rows`` deterministic
    rows per table) and persists it with SQLite's backup API. Writes to a temp file first so
    an interrupted run never leaves a half-written database at ``dest``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)

    mem_conn = build_sampled_connection(src, n_rows, seed=seed)
    try:
        disk_conn = sqlite3.connect(str(tmp))
        try:
            mem_conn.backup(disk_conn)
        finally:
            disk_conn.close()
    finally:
        mem_conn.close()

    tmp.replace(dest)
    return dest
