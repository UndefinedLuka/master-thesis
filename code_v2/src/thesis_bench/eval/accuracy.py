"""Execution-accuracy helpers and the canonical rows-to-text stringifier.

``rows_to_text`` is the single function that turns a query result (a set of rows)
into a deterministic string. It is used in three places — building the ground
truth, formatting a Text-to-SQL result for scoring, and any debugging — so the
text similarity metrics (ROUGE-L, BERTScore) compare like with like. Without one
canonical stringification, those metrics on tabular output would be noise.

Determinism: cells are rendered with ``cell_text`` (same as positional
serializers), columns keep query order, and rows are sorted by their rendered
tuple so two runs of the same query produce identical text regardless of row
order returned by SQLite.
"""

from __future__ import annotations

from ..serialization._values import cell_text
from ..serialization.snapshot import Row

# Delimiters chosen to be stable and unlikely to collide with cell content.
_CELL_SEP = " | "
_ROW_SEP = "\n"


def _row_key(row: Row, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(cell_text(row.get(col)) for col in columns)


def rows_to_text(columns: tuple[str, ...], rows: tuple[Row, ...]) -> str:
    """Canonical, order-independent string rendering of a result set."""
    if not rows:
        return ""
    rendered = sorted(_row_key(row, columns) for row in rows)
    return _ROW_SEP.join(_CELL_SEP.join(cells) for cells in rendered)


def row_set_match(
    expected_columns: tuple[str, ...],
    expected_rows: tuple[Row, ...],
    actual_columns: tuple[str, ...],
    actual_rows: tuple[Row, ...],
) -> bool:
    """Whether two result sets contain the same rows (ignoring row order).

    This is set-equality on the rendered rows, the standard notion of Text-to-SQL
    "execution accuracy" against a gold query. Column *names* may differ between
    the gold and generated queries (aliases), so we compare by position via each
    query's own column order.
    """
    expected = sorted(_row_key(r, expected_columns) for r in expected_rows)
    actual = sorted(_row_key(r, actual_columns) for r in actual_rows)
    return expected == actual
