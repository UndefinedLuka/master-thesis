"""Serializer round-trips and deterministic sub-sampling."""

from __future__ import annotations

from pathlib import Path

from benchmark.serialize import (
    SERIALIZERS,
    get_db_contents,
    get_db_schema,
    serialize,
)


def test_all_formats_handle_null_and_blob(tiny_db: Path) -> None:
    contents = get_db_contents(tiny_db, sample_n=None, seed=42)
    assert set(contents) == {"people", "pets"}
    # The blob must have been coerced to a placeholder string, NULL preserved.
    alice = contents["people"][0]
    assert isinstance(alice["photo"], str) and "BLOB" in alice["photo"]
    assert contents["people"][1]["age"] is None

    for fmt in SERIALIZERS:
        text = serialize(contents, fmt)
        assert text and isinstance(text, str), f"{fmt} produced empty output"
        # The table names should appear in every textual rendering.
        assert "people" in text and "pets" in text, f"{fmt} missing table names"


def test_markdown_has_header_and_separator(tiny_db: Path) -> None:
    contents = get_db_contents(tiny_db, sample_n=None, seed=42)
    md = serialize(contents, "markdown")
    assert "## people" in md
    assert "| --- |" in md  # GFM separator row present


def test_schema_dump(tiny_db: Path) -> None:
    schema = get_db_schema(tiny_db)
    assert "CREATE TABLE" in schema
    assert "people" in schema and "pets" in schema


def test_subsample_is_deterministic(tiny_db: Path) -> None:
    a = get_db_contents(tiny_db, sample_n=2, seed=7)
    b = get_db_contents(tiny_db, sample_n=2, seed=7)
    assert a == b
    assert len(a["pets"]) == 2  # 3 rows down-sampled to 2
    # A different seed may differ, but cardinality is stable.
    c = get_db_contents(tiny_db, sample_n=2, seed=99)
    assert len(c["pets"]) == 2
