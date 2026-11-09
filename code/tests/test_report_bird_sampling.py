"""Sampling sensitivity uses real in-memory SQLite samples and writes only a final summary."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from cli import report_bird_sampling as sampling
from config.paths import ProjectPaths
from config.settings import default_db_root, default_dev_json, sampled_db_root
from datasets.bird import load_bird_examples
from reference.answers import generate_reference_answer, write_reference_file
from serialization.tokens import TOKENIZER_FAMILIES
from sqlite.materialize import materialize_sampled_db


@pytest.fixture
def inputs(tmp_path, tiny_db, monkeypatch):
    paths = ProjectPaths.from_root(tmp_path)
    source = default_db_root("bird", paths.data_dir) / "tiny" / "tiny.sqlite"
    source.parent.mkdir(parents=True)
    shutil.copyfile(tiny_db, source)
    benchmark = sampled_db_root(paths.data_dir) / "tiny" / "tiny.sqlite"
    materialize_sampled_db(source, benchmark, 2, seed=sampling.SAMPLE_SEED)
    dev = default_dev_json("bird", paths.data_dir)
    queries = [
        "SELECT name FROM singer",
        "SELECT name FROM singer WHERE 0",
        "SELECT 0 AS value",
        "SELECT NULL AS value",
        "SELECT CAST(X'80' AS TEXT) AS value",
        "SELECT * FROM missing_table",
    ]
    dev.write_text(
        json.dumps(
            [
                {"db_id": "tiny", "question": sql, "SQL": sql, "difficulty": "simple"}
                for sql in queries
            ]
        )
    )
    examples = load_bird_examples(dev)
    write_reference_file(
        paths.data_dir / "bird_references.ndjson",
        [generate_reference_answer(example, benchmark) for example in examples],
    )
    paths.models_dir.mkdir()
    for family in TOKENIZER_FAMILIES:
        (paths.models_dir / family.preferred_model).write_bytes(b"fake GGUF")
    counters = []
    connections = []

    class Counter:
        def __init__(self, selection):
            self.family = selection.family
            self.model_path = selection.model_path
            self.closed = False

        def count_tokens(self, payload):
            return len(payload)

        def close(self):
            self.closed = True

    def create(selection):
        counter = Counter(selection)
        counters.append(counter)
        return counter

    build = sampling.build_sampled_connection

    def build_memory(*args, **kwargs):
        connection = build(*args, **kwargs)
        # An empty filename identifies a true in-memory SQLite database.
        assert connection.execute("PRAGMA database_list").fetchone()[2] == ""
        connections.append(connection)
        return connection

    monkeypatch.setattr(sampling, "BIRD_SAMPLE_ROWS", 2)
    monkeypatch.setattr(sampling.LlamaCppTokenCounter, "create", create)
    monkeypatch.setattr(sampling, "build_sampled_connection", build_memory)
    monkeypatch.setattr(ProjectPaths, "discover", lambda: paths)
    return paths, counters, connections


def _snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _assert_closed(counters, connections):
    assert len(counters) == 3 and all(counter.closed for counter in counters)
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_sensitivity_calculation_writes_no_files_and_preserves_baseline_checks(inputs):
    paths, counters, connections = inputs
    before = _snapshot(paths.root)
    report = sampling.calculate_bird_sampling(paths, row_caps=(1, 2, 3))
    assert report["total_questions"] == 6
    assert report["total_databases"] == 1
    sampled_rows = [sample["by_database"]["tiny"]["sampled_rows"] for sample in report["samples"]]
    assert sampled_rows == [2, 4, 5]
    for sample in report["samples"]:
        assert sample["empty_results"] == 1
        assert sample["non_empty_results"] == 4  # NULL, zero, and dirty text remain nonempty.
        assert sample["gold_sql_failures"] == 1
        assert sample["payload_measurements"] == 18
    assert report["samples"][1]["baseline_verification"] == {
        "databases_with_identical_contents": 1,
        "references_with_identical_error_and_empty_status": 6,
        "non_empty_row_count_changes": [],
        "identical_payload_token_counts": 18,
    }
    assert _snapshot(paths.root) == before
    assert not paths.results_dir.exists()
    _assert_closed(counters, connections)


def test_cli_writes_only_completed_summary(inputs, monkeypatch):
    paths, counters, connections = inputs
    before = _snapshot(paths.root)
    calculate = sampling.calculate_bird_sampling
    monkeypatch.setattr(
        sampling,
        "calculate_bird_sampling",
        lambda paths: calculate(
            paths,
            row_caps=(1, 2, 3),
        ),
    )
    assert sampling.main() == 0
    summary = Path("analysis/summaries/bird_sampling_sensitivity.json")
    after = _snapshot(paths.root)
    assert set(after) - set(before) == {summary}
    assert all(after[path] == content for path, content in before.items())
    assert len(json.loads(after[summary])["samples"]) == 3
    assert not paths.results_dir.exists()
    _assert_closed(counters, connections)


def test_failed_calculation_preserves_existing_summary_and_closes_resources(inputs, monkeypatch):
    paths, counters, connections = inputs
    refs_path = paths.data_dir / "bird_references.ndjson"
    references = [json.loads(line) for line in refs_path.read_text().splitlines()]
    references[0]["row_count"] = 0
    write_reference_file(refs_path, references)
    summary = paths.root / "analysis/summaries/bird_sampling_sensitivity.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"previous": true}\n')
    before = _snapshot(paths.root)
    calculate = sampling.calculate_bird_sampling
    monkeypatch.setattr(
        sampling,
        "calculate_bird_sampling",
        lambda paths: calculate(
            paths,
            row_caps=(1, 2),
        ),
    )
    with pytest.raises(ValueError, match="Reference empty/error status changed"):
        sampling.main()
    assert _snapshot(paths.root) == before
    _assert_closed(counters, connections)


def test_changed_benchmark_rows_are_rejected(inputs):
    paths, counters, connections = inputs
    benchmark = sampled_db_root(paths.data_dir) / "tiny" / "tiny.sqlite"
    with sqlite3.connect(benchmark) as connection:
        connection.execute("UPDATE singer SET name = 'changed'")
    with pytest.raises(ValueError, match="sample differs from benchmark"):
        sampling.calculate_bird_sampling(paths, row_caps=(2,))
    _assert_closed(counters, connections)
