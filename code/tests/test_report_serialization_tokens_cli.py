"""Tests for the standalone serialized-input token CSV report."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from cli.report_serialization_tokens import (
    DatabaseInput,
    DatasetInput,
    TokenizerFamily,
    TokenizerSelection,
    TokenRecord,
    _default_output_dir,
    discover_databases,
    load_token_records,
    measure_databases,
    resolve_tokenizer_models,
    write_token_report,
)
from config.paths import ProjectPaths

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


def _record(
    database_id: str,
    serialization_format: str,
    tokens: int,
    *,
    characters: int = 1_000,
) -> TokenRecord:
    return TokenRecord(
        dataset="spider",
        database_id=database_id,
        format=serialization_format,
        tokenizer_family=FAMILY.key,
        tokenizer_label=FAMILY.label,
        tokenizer_model=FAMILY.preferred_model,
        table_count=2,
        row_count=10,
        characters=characters,
        utf8_bytes=characters,
        tokens=tokens,
    )


def test_default_output_is_under_results_and_isolates_smoke_runs(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    selection = TokenizerSelection(FAMILY, tmp_path / FAMILY.preferred_model, ())
    full = _default_output_dir(
        paths,
        datasets=("spider", "bird"),
        formats=("json", "xml", "yaml", "csv", "markdown", "toon"),
        family_keys=("qwen-3.5", "ministral-3", "gemma-4"),
        limit=None,
        selections=(selection,),
        custom_tokenizers=False,
    )
    smoke = _default_output_dir(
        paths,
        datasets=("spider", "bird"),
        formats=("json", "xml", "yaml", "csv", "markdown", "toon"),
        family_keys=("qwen-3.5", "ministral-3", "gemma-4"),
        limit=1,
        selections=(selection,),
        custom_tokenizers=False,
    )

    assert full == tmp_path / "results" / "input_token_overhead"
    assert smoke.parent == full / "subsets"
    assert "limit-1" in smoke.name


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


def test_measure_databases_skips_completed_tokenizer_target(tiny_db: Path) -> None:
    counter = FakeCounter(FAMILY, Path(FAMILY.preferred_model))
    completed_record = _record("tiny", "json", 1)
    checkpointed: list[TokenRecord] = []

    records = measure_databases(
        "spider",
        [DatabaseInput("tiny", tiny_db)],
        ("json", "csv"),
        [counter],
        completed={
            (
                completed_record.dataset,
                completed_record.database_id,
                completed_record.format,
                completed_record.tokenizer_family,
                completed_record.tokenizer_model,
            )
        },
        on_record=checkpointed.append,
        show_progress=False,
    )

    assert [record.format for record in records] == ["csv"]
    assert checkpointed == records


def test_csv_report_round_trip_and_columns(tmp_path: Path) -> None:
    report = tmp_path / "report.csv"
    records = [_record("a", "json", 100), _record("a", "csv", 50)]

    write_token_report(report, records)

    assert load_token_records(report) == records
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "dataset",
        "database_id",
        "format",
        "tokenizer_family",
        "tokenizer_label",
        "tokenizer_model",
        "table_count",
        "row_count",
        "characters",
        "utf8_bytes",
        "tokens",
    ]
    assert rows[1]["tokens"] == "50"
    assert not (tmp_path / ".report.csv.tmp").exists()


def test_csv_report_can_checkpoint_an_empty_run(tmp_path: Path) -> None:
    report = tmp_path / "report.csv"

    write_token_report(report, [])

    assert load_token_records(report) == []
    assert report.read_text(encoding="utf-8").startswith("dataset,database_id,format,")


def test_dataset_input_carries_dev_cohort_size() -> None:
    dataset = DatasetInput(
        name="spider",
        example_count=1_034,
        databases=(DatabaseInput("singer", Path("singer.sqlite")),),
    )

    assert dataset.example_count == 1_034
