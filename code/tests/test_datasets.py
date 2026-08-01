"""Tests for the Spider/BIRD dev.json loaders and DB-path resolution."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import DatasetConfig
from datasets.bird import load_bird_examples
from datasets.loader import db_path_for, load_dataset, resolve_db_path
from datasets.spider import load_spider_examples


def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_spider_examples(tmp_path: Path) -> None:
    dev = _write_json(
        tmp_path / "dev.json",
        [
            {"db_id": "concert", "question": "How many singers?", "query": "SELECT count(*)"},
            {"db_id": "pets", "question": "How many pets?", "query": "SELECT count(*) FROM p"},
        ],
    )
    examples = load_spider_examples(dev)
    assert len(examples) == 2
    assert examples[0]["dataset"] == "spider"
    assert examples[0]["gold_sql"] == "SELECT count(*)"
    assert examples[0]["evidence"] is None
    assert examples[0]["example_id"] == "spider:concert:0"
    assert examples[1]["example_id"] == "spider:pets:1"


def test_load_bird_examples_with_key_aliases(tmp_path: Path) -> None:
    # BIRD releases have used SQL / query / sql for the gold query; all must load.
    dev = _write_json(
        tmp_path / "dev.json",
        [
            {"db_id": "a", "question": "q1", "SQL": "SELECT 1", "evidence": "hint"},
            {"db_id": "b", "question": "q2", "query": "SELECT 2"},
            {"db_id": "c", "question": "q3", "sql": "SELECT 3", "difficulty": "simple"},
        ],
    )
    examples = load_bird_examples(dev)
    assert [e["gold_sql"] for e in examples] == ["SELECT 1", "SELECT 2", "SELECT 3"]
    assert examples[0]["evidence"] == "hint"
    assert examples[1]["evidence"] is None
    assert examples[2]["difficulty"] == "simple"
    assert examples[0]["example_id"] == "bird:a:0"


def test_load_dataset_dispatches_on_name(tmp_path: Path) -> None:
    dev = _write_json(tmp_path / "dev.json", [{"db_id": "d", "question": "q", "query": "SELECT 1"}])
    cfg = DatasetConfig(name="spider", dev_json=dev, db_root=tmp_path)
    examples = load_dataset(cfg)
    assert examples[0]["dataset"] == "spider"


def test_db_path_layout(tmp_path: Path) -> None:
    cfg = DatasetConfig(name="spider", dev_json=tmp_path / "dev.json", db_root=tmp_path / "dbs")
    assert db_path_for("concert", cfg) == tmp_path / "dbs" / "concert" / "concert.sqlite"

    example = {"db_id": "concert"}
    assert resolve_db_path(example, cfg) == db_path_for("concert", cfg)  # type: ignore[arg-type]
