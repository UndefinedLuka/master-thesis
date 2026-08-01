"""Tests for the resumable NDJSON result store."""

from __future__ import annotations

from pathlib import Path

from config.settings import BackendConfig, DatasetConfig, RunConfig
from datasets.types import DatasetExample
from pipelines.shared import base_result
from pipelines.types import PipelineResult
from results.store import (
    ResultWriter,
    completed_key,
    load_completed_keys,
    load_results,
    result_dir,
    result_path,
    write_results,
)


def append_result(path: Path, result: PipelineResult) -> None:
    """Append via a fresh writer (tests exercise the durable-append path per record)."""
    ResultWriter(path).append(result)


def _run_config(tmp_path: Path, *, pipeline: str) -> RunConfig:
    return RunConfig(
        backend=BackendConfig(model_id="Qwen2.5-7B-Instruct-Q8_0.gguf"),
        dataset=DatasetConfig(name="spider", dev_json=Path(), db_root=Path()),
        pipeline=pipeline,  # type: ignore[arg-type]
        results_dir=tmp_path / "results",
    )


def _example(idx: int) -> DatasetExample:
    return DatasetExample(
        dataset="spider",
        db_id="db",
        question="q",
        gold_sql="SELECT 1",
        evidence=None,
        difficulty=None,
        example_id=f"spider:db:{idx}",
    )


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out.ndjson"
    r1 = base_result(_example(0), pipeline="direct", fmt="json", model_id="m")
    r2 = base_result(_example(1), pipeline="direct", fmt="markdown", model_id="m")
    append_result(path, r1)
    append_result(path, r2)

    loaded = load_results(path)
    assert len(loaded) == 2
    assert loaded[0]["example_id"] == "spider:db:0"


def test_completed_keys_distinguish_format(tmp_path: Path) -> None:
    path = tmp_path / "out.ndjson"
    append_result(path, base_result(_example(0), pipeline="direct", fmt="json", model_id="m"))
    append_result(path, base_result(_example(0), pipeline="direct", fmt="xml", model_id="m"))

    keys = load_completed_keys(path)
    assert ("spider:db:0", "direct", "json") in keys
    assert ("spider:db:0", "direct", "xml") in keys
    assert ("spider:db:0", "direct", "yaml") not in keys


def test_completed_key_for_text_to_sql_has_none_format() -> None:
    result = base_result(_example(0), pipeline="text_to_sql", fmt=None, model_id="m")
    assert completed_key(result) == ("spider:db:0", "text_to_sql", None)


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_results(tmp_path / "nope.ndjson") == []
    assert load_completed_keys(tmp_path / "nope.ndjson") == set()


def test_result_path_splits_by_dataset_pipeline_format(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path, pipeline="direct")
    json_path = result_path(cfg, "json")
    md_path = result_path(cfg, "markdown")

    # layout: <results_dir>/<dataset>/<pipeline>/<format>/<model>.ndjson
    assert json_path.parent == cfg.results_dir / "spider" / "direct" / "json"
    assert md_path.parent == cfg.results_dir / "spider" / "direct" / "markdown"
    assert json_path != md_path  # each format in its own folder/file
    assert json_path.name == "Qwen2-5-7B-Instruct-Q8_0-gguf.ndjson"


def test_result_path_text_to_sql_uses_schema_folder(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path, pipeline="text_to_sql")
    path = result_path(cfg, None)
    assert path.parent == cfg.results_dir / "spider" / "text_to_sql" / "schema"


def test_result_dir_matches_path_parent(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path, pipeline="direct")
    assert result_dir(cfg, "yaml") == result_path(cfg, "yaml").parent


# --- crash safety / resume robustness -----------------------------------------------------


def test_append_writes_complete_newline_terminated_line(tmp_path: Path) -> None:
    path = tmp_path / "out.ndjson"
    append_result(path, base_result(_example(0), pipeline="direct", fmt="json", model_id="m"))
    assert path.read_bytes().endswith(b"\n")


def test_append_repairs_partial_trailing_line(tmp_path: Path) -> None:
    # Simulate a process killed mid-write: a truncated final line with no newline.
    path = tmp_path / "out.ndjson"
    append_result(path, base_result(_example(0), pipeline="direct", fmt="json", model_id="m"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"example_id": "spider:db:1", "pipeli')  # truncated, no newline

    # The next durable append must not glue onto the partial line.
    append_result(path, base_result(_example(2), pipeline="direct", fmt="json", model_id="m"))

    lines = path.read_text().splitlines()
    assert lines[0].endswith("}")  # first record intact
    assert lines[1] == '{"example_id": "spider:db:1", "pipeli'  # truncated record isolated
    assert lines[2].endswith("}")  # new record on its own clean line


def test_load_results_skips_corrupt_line(tmp_path: Path, capsys) -> None:
    path = tmp_path / "out.ndjson"
    append_result(path, base_result(_example(0), pipeline="direct", fmt="json", model_id="m"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
    append_result(path, base_result(_example(2), pipeline="direct", fmt="json", model_id="m"))

    loaded = load_results(path)  # must not raise
    ids = {r["example_id"] for r in loaded}
    assert ids == {"spider:db:0", "spider:db:2"}  # corrupt middle line skipped
    assert "malformed" in capsys.readouterr().err


def test_completed_keys_survive_partial_crash(tmp_path: Path) -> None:
    # Two finished records + a truncated third (the crash point).
    path = tmp_path / "out.ndjson"
    append_result(path, base_result(_example(0), pipeline="direct", fmt="json", model_id="m"))
    append_result(path, base_result(_example(1), pipeline="direct", fmt="json", model_id="m"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"example_id": "spider:db:2", "pipeline": "dir')  # crashed here

    # Resume sees exactly the two completed examples; the unfinished one is re-run.
    keys = load_completed_keys(path)
    assert keys == {
        ("spider:db:0", "direct", "json"),
        ("spider:db:1", "direct", "json"),
    }
    # And appending the re-run of example 2 yields a clean, fully-loadable file.
    append_result(path, base_result(_example(2), pipeline="direct", fmt="json", model_id="m"))
    assert len(load_results(path)) == 3


def test_write_results_roundtrip_and_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "scored.ndjson"
    first = [base_result(_example(0), pipeline="direct", fmt="json", model_id="m")]
    second = [
        base_result(_example(1), pipeline="direct", fmt="json", model_id="m"),
        base_result(_example(2), pipeline="direct", fmt="json", model_id="m"),
    ]
    write_results(path, first)
    write_results(path, second)  # overwrites, no appending
    loaded = load_results(path)
    assert [r["example_id"] for r in loaded] == ["spider:db:1", "spider:db:2"]


def test_result_writer_fsync_cadence(tmp_path: Path, monkeypatch) -> None:
    import results.store as store_module

    fsyncs: list[int] = []
    monkeypatch.setattr(store_module.os, "fsync", lambda fd: fsyncs.append(fd))

    writer = ResultWriter(tmp_path / "out.ndjson", fsync_interval=2)
    for idx in range(5):
        writer.append(base_result(_example(idx), pipeline="direct", fmt="json", model_id="m"))
    # 5 appends at interval 2 -> fsync after records 2 and 4.
    assert len(fsyncs) == 2


def test_result_writers_do_not_share_state(tmp_path: Path, monkeypatch) -> None:
    import results.store as store_module

    fsyncs: list[int] = []
    monkeypatch.setattr(store_module.os, "fsync", lambda fd: fsyncs.append(fd))

    a = ResultWriter(tmp_path / "a.ndjson", fsync_interval=2)
    b = ResultWriter(tmp_path / "b.ndjson", fsync_interval=2)
    a.append(base_result(_example(0), pipeline="direct", fmt="json", model_id="m"))
    b.append(base_result(_example(0), pipeline="direct", fmt="json", model_id="m"))
    # One append each: neither writer reaches its own interval (no cross-writer counter).
    assert fsyncs == []
