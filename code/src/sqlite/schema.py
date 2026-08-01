"""Schema extraction: table names and CREATE TABLE DDL.

The concatenated DDL is exactly what the Text-to-SQL prompt provides to the model (schema,
not raw data), so the model must infer table contents from names and column types.
"""

from __future__ import annotations

from pathlib import Path

from sqlite.connection import query_sqlite


def get_table_names(db_path: Path | str) -> list[str]:
    """Names of all real tables (excludes views and sqlite_ internal tables)."""
    query = (
        "SELECT name FROM sqlite_master"
        " WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%';"
    )
    result = query_sqlite(db_path, query)
    if not result.rows:
        return []
    return [r["name"] for r in result.rows]


def get_schema_ddl(db_path: Path | str) -> str:
    """Concatenated CREATE TABLE statements, separated by blank lines."""
    query = (
        "SELECT sql FROM sqlite_master"
        " WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%';"
    )
    result = query_sqlite(db_path, query)
    if not result.rows:
        return ""
    statements = [r["sql"] for r in result.rows if r["sql"]]
    return "\n\n".join(statements)
