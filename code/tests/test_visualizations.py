"""Tests for shared visualization data loading and the input-token chart."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.average_input_tokens_by_format import (  # noqa: E402
    calculate_average_input_tokens,
    plot_average_input_tokens,
    save_figure,
)
from visualizations.data import (  # noqa: E402
    HAS_ANSWER_COLUMN,
    METHOD_LABELS,
    METRIC_TITLES,
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
        "Normalized Exact Match",
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
    summary = calculate_average_input_tokens(cohort.records).set_index("format")

    assert cohort.question_count == 1
    assert cohort.observation_count == 4
    assert set(cohort.records["example_id"]) == {"short"}
    assert summary.loc["csv", "answer_count"] == 2
    assert summary.loc["json", "answer_count"] == 2
    assert summary.loc["csv", "average_input_tokens"] == pytest.approx(100.0)
    assert summary.loc["json", "average_input_tokens"] == pytest.approx(100.0)


def test_calculate_average_input_tokens_groups_formats_and_sql_execution() -> None:
    results = pd.DataFrame(
        {
            "pipeline": ["direct", "direct", "direct", "direct", "direct", "text_to_sql"],
            "format": ["json", "csv", "json", "csv", "xml", None],
            "input_tokens": [100, 50, 300, 150, 400, 20],
        }
    )

    summary = calculate_average_input_tokens(results)
    indexed = summary.set_index("format")

    assert indexed.loc["csv", "average_input_tokens"] == pytest.approx(100.0)
    assert indexed.loc["json", "average_input_tokens"] == pytest.approx(200.0)
    assert indexed.loc["SQL execution", "average_input_tokens"] == pytest.approx(20.0)
    assert indexed.loc["csv", "answer_count"] == 2
    assert indexed.loc["json", "answer_count"] == 2
    assert summary["format"].tolist() == ["csv", "json", "xml", "SQL execution"]


def test_plot_can_be_saved(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "format": ["csv", "json", "xml", "SQL execution"],
            "average_input_tokens": [100.0, 200.0, 300.0, 20.0],
            "answer_count": [2, 2, 2, 2],
        }
    )

    figure, axes = plot_average_input_tokens(summary)
    output = tmp_path / "nested" / "tokens.png"
    save_figure(figure, output)

    assert axes.get_title() == "Average Input Tokens by Method"
    assert axes.get_xlabel() == "Method"
    assert [label.get_text() for label in axes.get_xticklabels()][-2:] == [
        "XML",
        "SQL",
    ]
    assert output.is_file()
    assert output.stat().st_size > 0
