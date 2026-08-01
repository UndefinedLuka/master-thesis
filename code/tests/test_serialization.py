"""Pure-function tests for the six serializers and the dispatcher."""

from __future__ import annotations

import json

import yaml

from serialization import FORMATS, serialize
from serialization.types import DatabaseContents

SAMPLE: DatabaseContents = {
    "singer": [
        {"id": 1, "name": "Alice", "age": 30, "note": None},
        {"id": 2, "name": "Bob | the second", "age": 25, "note": "x\ny"},
    ],
    "concert": [
        {"id": 10, "singer_id": 1, "city": "Zürich"},
    ],
}


def test_all_formats_dispatch_and_nonempty() -> None:
    for fmt in FORMATS:
        out = serialize(SAMPLE, fmt)
        assert isinstance(out, str)
        assert out.strip(), f"{fmt} produced empty output"


def test_json_roundtrip() -> None:
    out = serialize(SAMPLE, "json")
    assert json.loads(out) == SAMPLE


def test_yaml_roundtrip() -> None:
    out = serialize(SAMPLE, "yaml")
    assert yaml.safe_load(out) == SAMPLE


def test_yaml_preserves_unicode() -> None:
    assert "Zürich" in serialize(SAMPLE, "yaml")


def test_csv_has_headers_and_table_labels() -> None:
    out = serialize(SAMPLE, "csv")
    assert "# table: singer" in out
    assert "# table: concert" in out
    assert "id,name,age,note" in out


def test_markdown_has_pipe_tables_and_escapes_pipes() -> None:
    out = serialize(SAMPLE, "markdown")
    assert "## singer" in out
    assert "| --- |" in out or "| --- | --- |" in out
    # the literal pipe in "Bob | the second" must be escaped so the grid stays intact
    assert r"Bob \| the second" in out
    # newline in a cell must not break the row
    assert "x\ny" not in out


def test_toon_encodes() -> None:
    out = serialize(SAMPLE, "toon")
    assert "singer" in out


def test_empty_database() -> None:
    for fmt in FORMATS:
        # should not raise on an empty / table-only database
        serialize({"t": []}, fmt)
