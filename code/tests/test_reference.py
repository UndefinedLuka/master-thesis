"""Tests for reference-answer generation (gold SQL execution)."""

from __future__ import annotations

from pathlib import Path

from config.settings import DatasetConfig
from datasets.types import DatasetExample
from reference.answers import build_reference_set, generate_reference_answer
from sqlite.materialize import materialize_sampled_db


def _example(gold_sql: str, dataset: str = "spider") -> DatasetExample:
    return DatasetExample(
        dataset=dataset,
        db_id="tiny",
        question="q",
        gold_sql=gold_sql,
        evidence=None,
        difficulty=None,
        example_id=f"{dataset}:tiny:0",
    )


def test_reference_captures_rows(tiny_db: Path) -> None:
    ref = generate_reference_answer(_example("SELECT name FROM singer ORDER BY id"), tiny_db)
    assert ref["error"] is None
    assert ref["row_count"] == 5
    assert ref["rows"][0]["name"] == "Alice"
    assert ref["column_names"] == ["name"]


def test_reference_captures_error_without_raising(tiny_db: Path) -> None:
    ref = generate_reference_answer(_example("SELECT * FROM nope"), tiny_db)
    assert ref["rows"] is None
    assert ref["error"] is not None


def test_reference_from_materialized_sample(tiny_db: Path) -> None:
    sampled = materialize_sampled_db(
        tiny_db, tiny_db.parent / "sampled" / "tiny.sqlite", n_rows=2, seed=0
    )
    ref = generate_reference_answer(_example("SELECT count(*) AS n FROM singer"), sampled)
    assert ref["error"] is None
    assert ref["rows"] is not None
    # sampled 2 rows, so count should be 2 (not 5)
    assert ref["rows"][0]["n"] == 2


def test_build_reference_set_uses_configured_db_root(tiny_db: Path) -> None:
    # BIRD references run against cfg.db_root, which the CLI points at the materialized
    # sampled databases — build one in the <db_root>/<db_id>/<db_id>.sqlite layout.
    sampled_root = tiny_db.parent / "bird_sampled"
    materialize_sampled_db(tiny_db, sampled_root / "tiny" / "tiny.sqlite", n_rows=2, seed=0)
    cfg = DatasetConfig(
        name="bird",
        dev_json=tiny_db.parent / "dev.json",  # not read by build_reference_set
        db_root=sampled_root,
    )
    example = _example("SELECT count(*) AS n FROM singer", dataset="bird")
    refs = build_reference_set([example], cfg)
    assert refs[0]["error"] is None
    assert refs[0]["rows"] is not None
    assert refs[0]["rows"][0]["n"] == 2
