import sqlite3
from typing import Dict, List

import polars as pl


def query_sqlite(path: str, query: str) -> pl.DataFrame:
    conn = sqlite3.connect(path)

    conn.text_factory = lambda b: b.decode(errors="replace")  # type: ignore

    df: pl.DataFrame = pl.read_database(
        query=query, connection=conn, infer_schema_length=None
    )

    conn.close()

    return df


def get_all_table_names(path: str) -> List[str]:
    query = """select name from sqlite_master where type = "table" and sql not null;"""

    df = query_sqlite(path=path, query=query)

    return df["name"].to_list()


def get_foreign_keys(path: str, table_name: str) -> List[str]:
    query = f"PRAGMA foreign_key_list({table_name})"

    df = query_sqlite(path=path, query=query)

    return df["table"].to_list()


def get_db_foreign_keys(path: str) -> Dict[str, List[str]]:
    table_names = get_all_table_names(path=path)

    db_foreign_keys: Dict[str, List[str]] = {}

    for name in table_names:
        db_foreign_keys[name] = get_foreign_keys(path=path, table_name=name)

    return db_foreign_keys


def get_db_schema(path: str) -> List[str]:
    query = "SELECT * FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;"

    df = query_sqlite(path=path, query=query)

    return df["sql"].to_list()
