import sqlite3

import polars as pl


def query_sqlite(path: str, query: str) -> pl.DataFrame:
    conn = sqlite3.connect(path)

    conn.text_factory = lambda b: b.decode(errors="replace")  # type: ignore

    df: pl.DataFrame = pl.read_database(
        query=query, connection=conn, infer_schema_length=None
    )

    conn.close()

    return df
