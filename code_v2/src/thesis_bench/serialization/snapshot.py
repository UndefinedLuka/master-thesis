"""The in-memory database snapshot — the single source of truth for a run.

A ``DatabaseSnapshot`` captures the (possibly sampled) contents of one database,
once, so that:

1. all six serializers produce text from the *same* rows, and
2. the Text-to-SQL pipeline executes the model's query (and the reference query)
   against the *same* rows (via ``db.materialize_sample``).

This is what guarantees ch-03's "the same sample is used for both pipelines"
requirement for BIRD. For Spider the snapshot simply holds full tables.

Row values are kept as native Python scalars (``str``, ``int``, ``float``,
``bytes``, ``bool``, ``None``) so every serializer can decide how to render them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# A cell value as returned by sqlite3.
CellValue = str | int | float | bytes | bool | None

# One row keyed by column name. Declared as a ``Mapping`` (read-only, covariant in
# the value type) rather than ``dict`` so a concrete ``dict[str, int]`` row from a
# query is assignable wherever a ``Row`` is expected; we never mutate rows.
Row = Mapping[str, CellValue]


@dataclass(frozen=True, slots=True)
class TableSnapshot:
    """A single table's column order and rows.

    ``columns`` preserves the database column order, which positional formats
    (CSV, Markdown) and deterministic stringification rely on — ``Row`` dicts
    preserve insertion order but we never trust that; ``columns`` is canonical.
    """

    name: str
    columns: tuple[str, ...]
    rows: tuple[Row, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class DatabaseSnapshot:
    """A whole database captured in memory.

    Attributes:
        db_id: Database identifier (matches the dataset's ``db_id``).
        tables: Ordered tables. Order is the database's table order so output is
            stable across runs.
        schema_sql: The ``CREATE TABLE`` DDL (from ``sqlite_master``). Fed to the
            Text-to-SQL prompt; not used by the structural serializers.
        sample_rows: ``None`` when the full table was captured (Spider); the
            per-table cap ``n`` when sampled (BIRD). Recorded for provenance.
        sample_seed: The base seed used for sampling, ``None`` when not sampled.
    """

    db_id: str
    tables: tuple[TableSnapshot, ...]
    schema_sql: str
    sample_rows: int | None = None
    sample_seed: int | None = None

    def as_mapping(self) -> dict[str, list[Row]]:
        """Plain ``{table_name: [row, ...]}`` view for serializers that prefer it.

        This mirrors the structure v1's serializers consumed, easing the port.
        """
        return {t.name: list(t.rows) for t in self.tables}
