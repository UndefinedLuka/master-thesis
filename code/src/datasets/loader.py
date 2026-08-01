"""Unified dataset loading + database-path resolution."""

from __future__ import annotations

from pathlib import Path

from config.settings import DatasetConfig
from datasets.bird import load_bird_examples
from datasets.spider import load_spider_examples
from datasets.types import DatasetExample
from serialization.types import DatabaseContents
from sqlite.sampling import load_full_database


def load_dataset(cfg: DatasetConfig) -> list[DatasetExample]:
    """Load examples for the configured dataset."""
    match cfg.name:
        case "spider":
            return load_spider_examples(cfg.dev_json)
        case "bird":
            return load_bird_examples(cfg.dev_json)
        case _:  # pragma: no cover - guarded by DatasetName Literal
            raise ValueError(f"Unknown dataset: {cfg.name!r}")


def resolve_db_path(example: DatasetExample, cfg: DatasetConfig) -> Path:
    """Locate the SQLite file for an example's database.

    Both Spider and BIRD store databases as ``<db_root>/<db_id>/<db_id>.sqlite``; ``db_root``
    differs per dataset (``.../database`` vs ``.../dev_databases``) and is set in config.
    """
    return cfg.db_root / example["db_id"] / f"{example['db_id']}.sqlite"


def db_path_for(db_id: str, cfg: DatasetConfig) -> Path:
    """Locate the SQLite file for a database id (when iterating DBs, not examples)."""
    return cfg.db_root / db_id / f"{db_id}.sqlite"


def load_db_contents(db_path: Path) -> DatabaseContents:
    """Load every row of the database at ``db_path``.

    Spider points at the downloaded databases; BIRD points at the MATERIALIZED sampled
    copies (``db_root`` = ``data/bird_sampled``, built by thesis-sample-dbs), so both
    pipelines and the reference answers read the exact same rows.
    """
    return load_full_database(db_path)
