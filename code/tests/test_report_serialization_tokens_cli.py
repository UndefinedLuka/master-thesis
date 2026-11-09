"""Tests for in-memory token measurement and the stdout-only reporting command."""

from __future__ import annotations

import csv
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from cli import report_serialization_tokens as cli
from config.paths import ProjectPaths
from config.settings import default_dev_json, effective_db_root
from serialization import tokens as measurement
from serialization.tokens import (
    DatabaseInput,
    TokenizerFamily,
    discover_databases,
    measure_databases,
    measure_serialization_tokens,
    resolve_tokenizer_models,
)

FAMILY = TokenizerFamily(
    key="family",
    label="Family",
    model_glob="Family-*.gguf",
    preferred_model="Family-small.gguf",
)


@dataclass
class FakeCounter:
    family: TokenizerFamily
    model_path: Path
    divisor: int = 1
    closed: bool = False

    def count_tokens(self, text: str) -> int:
        return len(text) // self.divisor

    def close(self) -> None:
        self.closed = True


def test_discover_databases_uses_conventional_layout_and_sort_order(tmp_path: Path) -> None:
    root = tmp_path / "databases"
    for database_id in ("zeta", "alpha"):
        directory = root / database_id
        directory.mkdir(parents=True)
        (directory / f"{database_id}.sqlite").write_bytes(b"SQLite format 3\0")
    (root / "README.txt").write_text("ignored", encoding="utf-8")

    databases = discover_databases(root)

    assert [database.database_id for database in databases] == ["alpha", "zeta"]
    assert databases[0].path == root / "alpha" / "alpha.sqlite"


def test_discover_databases_can_select_only_dev_split_ids(tmp_path: Path) -> None:
    root = tmp_path / "databases"
    for database_id in ("dev", "train"):
        directory = root / database_id
        directory.mkdir(parents=True)
        (directory / f"{database_id}.sqlite").write_bytes(b"SQLite format 3\0")

    databases = discover_databases(root, ["dev", "dev"])

    assert [database.database_id for database in databases] == ["dev"]


def test_discover_databases_rejects_missing_requested_database(tmp_path: Path) -> None:
    root = tmp_path / "databases"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="missing.sqlite"):
        discover_databases(root, ["missing"])


def test_resolve_tokenizer_models_prefers_configured_representative(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    preferred = models / FAMILY.preferred_model
    preferred.write_bytes(b"larger preferred model")
    (models / "Family-other.gguf").write_bytes(b"x")

    selection = resolve_tokenizer_models(models, [FAMILY])[0]

    assert selection.model_path == preferred.resolve()
    assert selection.covered_models == ("Family-other.gguf", "Family-small.gguf")


def test_resolve_tokenizer_models_uses_smallest_available_fallback(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "Family-large.gguf").write_bytes(b"large")
    smallest = models / "Family-tiny.gguf"
    smallest.write_bytes(b"x")

    selection = resolve_tokenizer_models(models, [FAMILY])[0]

    assert selection.model_path == smallest.resolve()


def test_measure_databases_counts_each_format_and_tokenizer(tiny_db: Path) -> None:
    counters = [
        FakeCounter(FAMILY, Path("one.gguf"), divisor=1),
        FakeCounter(
            TokenizerFamily("other", "Other", "Other-*.gguf", "Other-small.gguf"),
            Path("two.gguf"),
            divisor=2,
        ),
    ]

    records = measure_databases(
        "spider",
        [DatabaseInput("tiny", tiny_db)],
        ("json", "csv"),
        counters,
        show_progress=False,
    )

    assert len(records) == 4
    assert {(record.format, record.tokenizer_family) for record in records} == {
        ("json", "family"),
        ("json", "other"),
        ("csv", "family"),
        ("csv", "other"),
    }
    for serialization_format in ("json", "csv"):
        family_record = next(
            record
            for record in records
            if record.format == serialization_format and record.tokenizer_family == "family"
        )
        other_record = next(
            record
            for record in records
            if record.format == serialization_format and record.tokenizer_family == "other"
        )
        assert family_record.characters == family_record.tokens
        assert other_record.tokens == family_record.characters // 2
        assert family_record.table_count == 2
        assert family_record.row_count == 7


@pytest.fixture
def inputs(tmp_path, tiny_db, monkeypatch):
    paths = ProjectPaths.from_root(tmp_path)
    for dataset in ("spider", "bird"):
        dev = default_dev_json(dataset, paths.data_dir)
        dev.parent.mkdir(parents=True, exist_ok=True)
        query_key = "query" if dataset == "spider" else "SQL"
        dev.write_text(
            json.dumps(
                [
                    {"db_id": db, "question": "Count?", query_key: "SELECT count(*) FROM singer"}
                    for db in ("b", "a", "a")
                ]
            )
        )
        for db in ("a", "b", "unused"):
            dest = effective_db_root(dataset, paths.data_dir) / db / f"{db}.sqlite"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tiny_db, dest)
    paths.models_dir.mkdir()
    for family in measurement.TOKENIZER_FAMILIES:
        (paths.models_dir / family.preferred_model).write_bytes(b"fake GGUF")
    counters = []

    def create(selection):
        counter = FakeCounter(selection.family, selection.model_path)
        counters.append(counter)
        return counter

    monkeypatch.setattr(measurement.LlamaCppTokenCounter, "create", create)
    monkeypatch.setattr(ProjectPaths, "discover", lambda: paths)
    return paths, counters


def _files(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_measurement_reads_only_dev_databases_and_writes_nothing(inputs):
    paths, counters = inputs
    before = _files(paths.root)
    records = measure_serialization_tokens(paths, formats=("csv", "json"), show_progress=False)
    assert len(records) == 2 * 2 * 2 * 3
    assert {record.database_id for record in records} == {"a", "b"}
    assert {record.dataset for record in records} == {"spider", "bird"}
    assert all(record.row_count == 7 and record.tokens > 0 for record in records)
    assert len(counters) == 3
    assert all(counter.closed for counter in counters)
    assert _files(paths.root) == before
    assert not paths.results_dir.exists()


def test_measurement_closes_tokenizers_on_failure(inputs, monkeypatch):
    paths, counters = inputs
    before = _files(paths.root)

    def fail(self, text):
        raise RuntimeError("tokenizer failure")

    monkeypatch.setattr(FakeCounter, "count_tokens", fail)
    with pytest.raises(RuntimeError, match="Failed to tokenize"):
        measure_serialization_tokens(paths, show_progress=False)
    assert len(counters) == 3
    assert all(counter.closed for counter in counters)
    assert _files(paths.root) == before


def test_cli_prints_filtered_measurements_without_files(inputs, capsys):
    paths, counters = inputs
    before = _files(paths.root)
    assert (
        cli.main(
            [
                "--dataset",
                "bird",
                "--format",
                "csv",
                "--family",
                "qwen-3.5",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "bird"
    assert rows[0]["database_id"] == "a"
    assert rows[0]["format"] == "csv"
    assert rows[0]["tokenizer_family"] == "qwen-3.5"
    assert int(rows[0]["tokens"]) > 0
    assert counters[0].closed
    assert _files(paths.root) == before


@pytest.mark.parametrize("args", [["--output-dir", "output"], ["--no-resume"]])
def test_cli_has_no_persistence_options(args):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(args)
