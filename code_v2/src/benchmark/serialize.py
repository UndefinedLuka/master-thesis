"""Database serialization into the six study formats plus the SQL schema.

Ported and cleaned from the prototype ``convert_data.py``: TOML is dropped (not
one of the six study formats) and Markdown is added. The DB contents are built
*once* per database and reused for every format, which guarantees the BIRD
sub-sample is byte-for-byte identical across formats and across both pipelines.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Callable

import yaml
from dict2xml import dict2xml

from benchmark.sqlite_io import query_sqlite, to_records

# Format -> file extension used on disk.
EXT: dict[str, str] = {
    "json": "json",
    "yaml": "yaml",
    "xml": "xml",
    "toon": "toon",
    "csv": "csv",
    "markdown": "md",
}

Contents = dict[str, list[dict[str, Any]]]


# --------------------------------------------------------------------------- #
# Schema and table introspection
# --------------------------------------------------------------------------- #
def get_table_names(db_path: str | Path) -> list[str]:
    df = query_sqlite(
        db_path,
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%';",
    )
    return df["name"].to_list() if df.height else []


def get_db_schema(db_path: str | Path) -> str:
    """Return the concatenated ``CREATE TABLE`` statements for the database."""
    df = query_sqlite(
        db_path,
        "SELECT sql FROM sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%';",
    )
    return "\n\n".join(s for s in df["sql"].to_list() if s)


def _table_seed(seed: int, table: str) -> int:
    """Derive a stable per-table seed (stdlib ``hash`` is salted per process)."""
    h = hashlib.sha256(f"{seed}:{table}".encode()).hexdigest()
    return int(h[:16], 16)


def _sample_table(db_path: str | Path, table: str, sample_n: int, seed: int) -> list[dict]:
    """Return up to ``sample_n`` rows from ``table`` deterministically.

    Prefers rowid-based selection (cheap, avoids loading the whole table). Falls
    back to loading all rows then sampling in Python for WITHOUT ROWID tables.
    """
    quoted = '"' + table.replace('"', '""') + '"'
    rng = random.Random(_table_seed(seed, table))
    try:
        rowids = query_sqlite(db_path, f"SELECT rowid AS _rid FROM {quoted};")["_rid"].to_list()
        if len(rowids) <= sample_n:
            chosen = rowids
        else:
            chosen = rng.sample(rowids, sample_n)
        if not chosen:
            return []
        in_list = ",".join(str(int(r)) for r in chosen)
        df = query_sqlite(db_path, f"SELECT * FROM {quoted} WHERE rowid IN ({in_list});")
        return to_records(df)
    except Exception:
        # WITHOUT ROWID table (or no rowid): load all and sample the records.
        records = to_records(query_sqlite(db_path, f"SELECT * FROM {quoted};"))
        if len(records) <= sample_n:
            return records
        return rng.sample(records, sample_n)


def get_db_contents(db_path: str | Path, sample_n: int | None, seed: int) -> Contents:
    """Return ``{table: [row, ...]}`` for every table.

    When ``sample_n`` is set (BIRD), each table is deterministically sub-sampled
    to at most ``sample_n`` rows; otherwise (Spider) full tables are used.
    """
    contents: Contents = {}
    for table in get_table_names(db_path):
        if sample_n and sample_n > 0:
            contents[table] = _sample_table(db_path, table, sample_n, seed)
        else:
            quoted = '"' + table.replace('"', '""') + '"'
            contents[table] = to_records(query_sqlite(db_path, f"SELECT * FROM {quoted};"))
    return contents


# --------------------------------------------------------------------------- #
# Serializers
# --------------------------------------------------------------------------- #
def _to_json(contents: Contents) -> str:
    return json.dumps(contents, indent=2, ensure_ascii=False)


def _to_yaml(contents: Contents) -> str:
    return yaml.safe_dump(contents, sort_keys=False, allow_unicode=True)


def _to_xml(contents: Contents) -> str:
    return dict2xml(contents, wrap="database", indent="  ")


def _to_toon(contents: Contents) -> str:
    import toon

    return toon.encode(contents)


def _to_csv(contents: Contents) -> str:
    """One CSV block per table, each prefixed with a ``# table: <name>`` header.

    A single string is produced (the model receives the whole database at once),
    unlike the prototype which wrote one file per table.
    """
    parts: list[str] = []
    for table, rows in contents.items():
        buf = io.StringIO()
        buf.write(f"# table: {table}\n")
        if rows:
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
        parts.append(buf.getvalue().rstrip("\n"))
    return "\n\n".join(parts) + "\n"


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _to_markdown(contents: Contents) -> str:
    """GitHub-flavored Markdown table per database table."""
    parts: list[str] = []
    for table, rows in contents.items():
        block = [f"## {table}", ""]
        if rows:
            cols = list(rows[0].keys())
            block.append("| " + " | ".join(_md_cell(c) for c in cols) + " |")
            block.append("| " + " | ".join("---" for _ in cols) + " |")
            for r in rows:
                block.append("| " + " | ".join(_md_cell(r.get(c)) for c in cols) + " |")
        else:
            block.append("*(no rows)*")
        parts.append("\n".join(block))
    return "\n\n".join(parts) + "\n"


SERIALIZERS: dict[str, Callable[[Contents], str]] = {
    "json": _to_json,
    "yaml": _to_yaml,
    "xml": _to_xml,
    "toon": _to_toon,
    "csv": _to_csv,
    "markdown": _to_markdown,
}


def serialize(contents: Contents, fmt: str) -> str:
    try:
        return SERIALIZERS[fmt](contents)
    except KeyError:
        raise ValueError(f"Unknown serialization format: {fmt!r}") from None


# --------------------------------------------------------------------------- #
# Writing / reading serialized databases on disk
# --------------------------------------------------------------------------- #
def serialize_db_all_formats(
    db_path: str | Path,
    out_dir: Path,
    formats: tuple[str, ...],
    sample_n: int | None,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Serialize one database to every format + schema; return per-artifact info.

    Returns ``{fmt_or_'sql': {'path': str, 'chars': int}}`` for the manifest.
    """
    db_path = Path(db_path)
    db_id = db_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    contents = get_db_contents(db_path, sample_n, seed)
    info: dict[str, dict[str, Any]] = {}

    schema = get_db_schema(db_path)
    schema_path = out_dir / f"{db_id}.sql"
    schema_path.write_text(schema, encoding="utf-8")
    info["sql"] = {"path": str(schema_path), "chars": len(schema)}

    for fmt in formats:
        text = serialize(contents, fmt)
        path = out_dir / f"{db_id}.{EXT[fmt]}"
        path.write_text(text, encoding="utf-8")
        info[fmt] = {"path": str(path), "chars": len(text)}

    return info


def serialized_path(out_dir: Path, db_id: str, fmt: str) -> Path:
    ext = "sql" if fmt == "sql" else EXT[fmt]
    return out_dir / f"{db_id}.{ext}"


def read_serialized(out_dir: Path, db_id: str, fmt: str) -> str:
    return serialized_path(out_dir, db_id, fmt).read_text(encoding="utf-8")
