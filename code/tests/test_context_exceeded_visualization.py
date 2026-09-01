"""Tests for the dataset-split context-exceeded visualization."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.context_exceeded_by_format import (  # noqa: E402
    calculate_context_exceeded_counts,
    plot_context_exceeded_counts,
    save_figure,
)


def test_counts_unique_questions_per_dataset_and_format() -> None:
    results = pd.DataFrame(
        {
            "example_id": [
                "bird:db:1",
                "bird:db:1",
                "bird:db:2",
                "bird:db:3",
                "spider:db:1",
                "spider:db:1",
                "spider:db:2",
                "bird:db:4",
            ],
            "dataset": [
                "bird",
                "bird",
                "bird",
                "bird",
                "spider",
                "spider",
                "spider",
                "bird",
            ],
            "pipeline": [
                "direct",
                "direct",
                "direct",
                "direct",
                "direct",
                "direct",
                "direct",
                "text_to_sql",
            ],
            "format": ["csv", "csv", "csv", "json", "csv", "csv", "json", None],
            "error": [
                "context_exceeded",
                "context_exceeded",
                "context_exceeded",
                None,
                "context_exceeded",
                "other_error",
                "context_exceeded",
                "context_exceeded",
            ],
        }
    )

    summary = calculate_context_exceeded_counts(
        results,
        datasets=("bird", "spider"),
        formats=("csv", "json", "xml"),
    )
    counts = summary.set_index(["format", "dataset"])["question_count"]

    # bird:db:1 appears for two model runs but is one question.
    assert counts.loc["csv", "bird"] == 2
    assert counts.loc["csv", "spider"] == 1
    assert counts.loc["json", "bird"] == 0
    assert counts.loc["json", "spider"] == 1
    assert counts.loc["xml", "bird"] == 0
    assert counts.loc["xml", "spider"] == 0
    assert list(summary[["format", "dataset"]].itertuples(index=False, name=None)) == [
        ("csv", "bird"),
        ("csv", "spider"),
        ("json", "bird"),
        ("json", "spider"),
        ("xml", "bird"),
        ("xml", "spider"),
    ]


def test_context_exceeded_plot_is_stacked_and_can_be_saved(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "format": ["csv", "csv", "xml", "xml"],
            "dataset": ["bird", "spider", "bird", "spider"],
            "question_count": [10, 3, 8, 0],
        }
    )

    figure, axes = plot_context_exceeded_counts(summary)
    output = tmp_path / "nested" / "context-exceeded.png"
    save_figure(figure, output)

    assert axes.get_title() == "Unique Questions Exceeding Context by Method"
    assert axes.get_xlabel() == "Method"
    assert [label.get_text() for label in axes.get_xticklabels()] == ["CSV", "XML"]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["BIRD", "Spider"]
    assert len(axes.containers) == 2
    assert {"10", "3", "8", "0"}.issubset(text.get_text() for text in axes.texts)
    assert output.is_file()
    assert output.stat().st_size > 0
