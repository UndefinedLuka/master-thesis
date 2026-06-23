"""Each serializer renders the fixture snapshot and carries the data through."""

from __future__ import annotations

import json

import pytest
import yaml

from thesis_bench.enums import SerializationFormat
from thesis_bench.serialization import registry
from thesis_bench.serialization.snapshot import DatabaseSnapshot


@pytest.mark.parametrize("fmt", list(SerializationFormat))
def test_serializer_runs_and_mentions_data(
    fmt: SerializationFormat, people_snapshot: DatabaseSnapshot
) -> None:
    out = registry.serialize(fmt, people_snapshot)
    assert isinstance(out, str)
    assert out
    # Every format must surface a known value somewhere in its output.
    assert "Alice" in out
    assert "people" in out


def test_json_roundtrips(people_snapshot: DatabaseSnapshot) -> None:
    out = registry.serialize(SerializationFormat.JSON, people_snapshot)
    data = json.loads(out)
    assert [r["name"] for r in data["people"]] == ["Alice", "Bob", "Carol"]


def test_yaml_roundtrips(people_snapshot: DatabaseSnapshot) -> None:
    out = registry.serialize(SerializationFormat.YAML, people_snapshot)
    data = yaml.safe_load(out)
    assert data["people"][0]["age"] == 30


def test_csv_has_header_once_per_table(people_snapshot: DatabaseSnapshot) -> None:
    out = registry.serialize(SerializationFormat.CSV, people_snapshot)
    assert out.count("id,name,age") == 1


def test_markdown_escapes_pipes() -> None:
    from thesis_bench.serialization.snapshot import TableSnapshot

    snap = DatabaseSnapshot(
        db_id="d",
        tables=(
            TableSnapshot(
                name="t", columns=("c",), rows=({"c": "a|b"},)
            ),
        ),
        schema_sql="",
    )
    out = registry.serialize(SerializationFormat.MARKDOWN, snap)
    assert "a\\|b" in out
