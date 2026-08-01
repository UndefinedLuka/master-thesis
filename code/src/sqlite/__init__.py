"""SQLite access: the only package that opens ``.sqlite`` files.

Querying via stdlib sqlite3; schema extraction yields CREATE TABLE DDL; sampling provides
BIRD sub-sampling and Spider full-table loading as plain ``DatabaseContents``.
"""

from sqlite.connection import (
    open_connection,
    query_connection,
    query_connection_safe,
    query_sqlite,
    query_sqlite_safe,
)
from sqlite.sampling import build_sampled_connection, load_full_database, sample_database
from sqlite.schema import get_schema_ddl, get_table_names

__all__ = [
    "open_connection",
    "query_sqlite",
    "query_sqlite_safe",
    "query_connection",
    "query_connection_safe",
    "get_table_names",
    "get_schema_ddl",
    "sample_database",
    "load_full_database",
    "build_sampled_connection",
]
