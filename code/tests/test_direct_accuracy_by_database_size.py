"""Tests for metric-specific, six-format database-size accuracy figures."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd

from visualizations.config import (
    DIRECT_ACCURACY_METRIC_ORDER,
    DIRECT_DATABASE_SIZE_FIGURE_FILENAMES,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.direct_accuracy_by_database_size import (
    calculate_database_size_accuracy,
    generate_database_size_accuracy_figures,
    plot_database_size_accuracy_metric,
)


def _result(
    *,
    dataset: str,
    db_id: str,
    score: float,
    model: str,
    fmt: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "db_id": db_id,
        "pipeline": "direct",
        "format": fmt,
        "model_id": model,
        "example_id": f"{dataset}:{db_id}:0",
        "scores.normalized_exact_match": score,
        "scores.bertscore_f": score - 0.1,
        "scores.rouge_l_f": score + 0.1,
    }


def _token_record(
    *,
    dataset: str,
    db_id: str,
    tokens: int,
    fmt: str,
    family: str,
    rows: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "database_id": db_id,
        "format": fmt,
        "tokenizer_family": family,
        "tokenizer_label": family,
        "tokenizer_model": f"{family}.gguf",
        "table_count": 2,
        "row_count": rows,
        "tokens": tokens,
    }


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, object]] = []
    report: list[dict[str, object]] = []
    databases = (
        ("spider", "small_spider", 0.8, 20, 100),
        ("spider", "large_spider", 0.2, 2_000, 10_000),
        ("bird", "small_bird", 0.6, 1_000, 1_000),
        ("bird", "large_bird", 0.1, 3_000, 20_000),
    )
    for dataset, db_id, score, rows, token_base in databases:
        for model in ("Qwen3.5-2B-Q4.gguf", "gemma-4-E4B-Q4.gguf"):
            for format_index, fmt in enumerate(SERIALIZATION_FORMAT_ORDER, start=1):
                results.append(
                    _result(
                        dataset=dataset,
                        db_id=db_id,
                        score=score,
                        model=model,
                        fmt=fmt,
                    )
                )
                if model.startswith("Qwen"):
                    for family_index, family in enumerate(("qwen", "gemma"), start=1):
                        report.append(
                            _token_record(
                                dataset=dataset,
                                db_id=db_id,
                                tokens=token_base * format_index * family_index,
                                fmt=fmt,
                                family=family,
                                rows=rows,
                            )
                        )
    return pd.DataFrame(results), pd.DataFrame(report)


def test_analysis_preserves_format_specific_size_and_accuracy() -> None:
    results, report = _inputs()
    analysis = calculate_database_size_accuracy(
        results,
        report,
        expected_database_counts={"spider": 2, "bird": 2},
    )
    scores = analysis.database_scores.set_index(["dataset", "db_id", "format", "metric"])

    csv_row = scores.loc[
        ("spider", "small_spider", "csv", "normalized_exact_match")
    ]
    json_row = scores.loc[
        ("spider", "small_spider", "json", "normalized_exact_match")
    ]
    assert csv_row["mean_score"] == pytest.approx(0.8)
    assert csv_row["mean_serialized_tokens"] == pytest.approx(150.0)
    assert json_row["mean_serialized_tokens"] == pytest.approx(750.0)
    assert csv_row["model_count"] == 2
    assert csv_row["observation_count"] == 2
    assert len(analysis.correlations) == 2 * 6 * 3
    assert all(analysis.correlations["pearson_log_size"] < 0)


def test_analysis_rejects_missing_database_format_size() -> None:
    results, report = _inputs()
    report = report[
        ~(
            report["database_id"].eq("large_bird")
            & report["format"].eq("xml")
        )
    ]

    with pytest.raises(ValueError, match="Missing serialization size"):
        calculate_database_size_accuracy(
            results,
            report,
            expected_database_counts={"spider": 2, "bird": 2},
        )


def test_each_metric_plot_has_six_format_panels_and_both_datasets(tmp_path: Path) -> None:
    results, report = _inputs()
    analysis = calculate_database_size_accuracy(
        results,
        report,
        expected_database_counts={"spider": 2, "bird": 2},
    )

    figures = [
        plot_database_size_accuracy_metric(analysis, metric)
        for metric in DIRECT_ACCURACY_METRIC_ORDER
    ]
    for (figure, axes), metric in zip(
        figures, DIRECT_ACCURACY_METRIC_ORDER, strict=True
    ):
        assert len(axes) == 6
        assert [axis.get_title() for axis in axes] == [
            "CSV",
            "TOON",
            "Markdown",
            "YAML",
            "JSON",
            "XML",
        ]
        assert all(axis.get_xscale() == "log" for axis in axes)
        assert all(len(axis.collections) == 2 for axis in axes)
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Spider",
            "BIRD",
        ]
        assert metric in DIRECT_DATABASE_SIZE_FIGURE_FILENAMES

    outputs = generate_database_size_accuracy_figures(
        results,
        report,
        tmp_path,
        excluded_model_markers=(),
        expected_database_counts={"spider": 2, "bird": 2},
    )
    assert outputs == [
        tmp_path / DIRECT_DATABASE_SIZE_FIGURE_FILENAMES[metric]
        for metric in DIRECT_ACCURACY_METRIC_ORDER
    ]
    assert all(path.stat().st_size > 0 for path in outputs)

    import matplotlib.pyplot as plt

    for figure, _ in figures:
        plt.close(figure)
