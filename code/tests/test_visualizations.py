"""Tests for shared visualization data loading and filtering utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

import pandas as pd

from visualizations.config import METHOD_LABELS, METRIC_TITLES
from visualizations.data import (
    HAS_ANSWER_COLUMN,
    discover_scored_files,
    load_scored_results,
    select_complete_question_cohort,
)


def test_shared_display_labels_match_publication_style() -> None:
    assert list(METHOD_LABELS.values()) == [
        "CSV",
        "Markdown",
        "TOON",
        "YAML",
        "JSON",
        "XML",
        "SQL",
    ]
    assert list(METRIC_TITLES.values()) == [
        "NEM",
        "ROUGE-L F1",
        "BERTScore F1",
    ]


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(
    *,
    example_id: str,
    fmt: str | None,
    input_tokens: int,
    model: str = "model.gguf",
    error: str | None = None,
    raw_output: str = "answer",
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "dataset": "bird",
        "pipeline": "direct",
        "format": fmt,
        "model_id": model,
        "input_tokens": input_tokens,
        "raw_output": raw_output,
        "error": error,
        "scores": {"normalized_exact_match": 1.0},
    }


def test_discover_scored_files_ignores_unscored_siblings(tmp_path: Path) -> None:
    scored = tmp_path / "bird" / "direct" / "json" / "model.scored.ndjson"
    unscored = scored.with_name("model.ndjson")
    _write_records(scored, [_record(example_id="1", fmt="json", input_tokens=10)])
    _write_records(unscored, [_record(example_id="1", fmt="json", input_tokens=10)])

    assert discover_scored_files(tmp_path) == [scored]


def test_load_scored_results_projects_filters_and_keeps_answers_only(tmp_path: Path) -> None:
    path = tmp_path / "results.scored.ndjson"
    _write_records(
        path,
        [
            _record(example_id="ok", fmt="json", input_tokens=100),
            _record(
                example_id="failed",
                fmt="json",
                input_tokens=0,
                error="context_exceeded",
                raw_output="",
            ),
            _record(example_id="other-model", fmt="yaml", input_tokens=60, model="other.gguf"),
        ],
    )

    loaded = load_scored_results(
        path,
        columns=("example_id", "format", "input_tokens", "scores.normalized_exact_match"),
        formats={"json"},
        models={"model.gguf"},
        answers_only=True,
    )

    assert loaded.to_dict("records") == [
        {
            "example_id": "ok",
            "format": "json",
            "input_tokens": 100,
            "scores.normalized_exact_match": 1.0,
        }
    ]


def test_load_scored_results_skips_malformed_lines(tmp_path: Path, capsys) -> None:
    path = tmp_path / "results.scored.ndjson"
    path.write_text(
        json.dumps(_record(example_id="ok", fmt="csv", input_tokens=42)) + "\n{broken\n",
        encoding="utf-8",
    )

    loaded = load_scored_results(path, columns=("input_tokens",))

    assert loaded["input_tokens"].tolist() == [42]
    assert "skipping line 2" in capsys.readouterr().err


def test_complete_cohort_removes_a_question_from_every_model_and_method() -> None:
    rows: list[dict[str, object]] = []
    for example_id, tokens in (("short", 100), ("long", 1_000)):
        for model_id in ("model-a", "model-b"):
            for method in ("csv", "json"):
                failed = example_id == "long" and model_id == "model-b" and method == "json"
                rows.append(
                    {
                        "dataset": "bird",
                        "example_id": example_id,
                        "model_id": model_id,
                        "pipeline": "direct",
                        "format": method,
                        "error": "context_exceeded" if failed else None,
                        HAS_ANSWER_COLUMN: not failed,
                        "input_tokens": 0 if failed else tokens,
                    }
                )

    cohort = select_complete_question_cohort(pd.DataFrame(rows))

    assert cohort.question_count == 1
    assert cohort.observation_count == 4
    assert set(cohort.records["example_id"]) == {"short"}
    assert set(cohort.records["input_tokens"]) == {100}
