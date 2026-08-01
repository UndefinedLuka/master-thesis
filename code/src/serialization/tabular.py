"""Helpers shared by the positional (tabular) serializers: CSV and Markdown."""

from __future__ import annotations

from serialization.types import Row


def union_columns(rows: list[Row]) -> list[str]:
    """All column names across ``rows``, in first-seen order.

    Rows may omit keys; the union preserves every column so no cell is silently dropped.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns
