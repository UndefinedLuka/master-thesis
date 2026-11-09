"""The serialization audit consumes in-memory measurements without an intermediate CSV."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pandas")

from cli import audit_serialization_tokens as audit
from config.paths import ProjectPaths
from serialization.tokens import TOKENIZER_FAMILIES, TokenRecord
from serialization.types import FORMATS


def _records():
    return [
        TokenRecord(
            dataset=dataset,
            database_id=db,
            format=fmt,
            tokenizer_family=family.key,
            tokenizer_label=family.label,
            tokenizer_model=family.preferred_model,
            table_count=1,
            row_count=1,
            characters=100,
            utf8_bytes=100,
            tokens=100 if fmt == "csv" else 200,
        )
        for dataset, databases in (
            ("spider", ["wta_1", *(f"s{i}" for i in range(19))]),
            ("bird", [f"b{i}" for i in range(11)]),
        )
        for db in databases
        for fmt in FORMATS
        for family in TOKENIZER_FAMILIES
    ]


def test_audit_calculations_and_hash_are_independent_of_disk_and_record_order():
    records = _records()
    report = audit.audit_serialization(records)
    assert report["record_count"] == 558
    assert report["database_counts"] == {"spider": 20, "bird": 11}
    assert report["csv_smallest_for_every_database_tokenizer"]
    assert report["wta_1_spider_csv"]["share_pct"] == 5
    assert (
        report["measurements_sha256"]
        == audit.audit_serialization(
            list(reversed(records)),
        )["measurements_sha256"]
    )
    for row in report["format_means_and_ratios"]:
        assert row["ratio_of_means_to_csv"] == (1 if row["format"] == "csv" else 2)
    with pytest.raises(ValueError, match="Duplicate"):
        audit.audit_serialization([*records, records[0]])
    with pytest.raises(ValueError, match="Incomplete"):
        audit.audit_serialization(records[:-1])


def test_cli_measures_in_memory_and_writes_only_final_audit(tmp_path, monkeypatch):
    paths = ProjectPaths.from_root(tmp_path)
    monkeypatch.setattr(ProjectPaths, "discover", lambda: paths)
    monkeypatch.setattr(audit, "measure_serialization_tokens", lambda *args, **kwargs: _records())
    audit.main([])
    output = tmp_path / "analysis/summaries/serialization_token_audit.json"
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [output]
    assert json.loads(output.read_text())["record_count"] == 558
