"""Tests for accuracy-by-model-size and family visualizations."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.accuracy_by_model_size import (  # noqa: E402
    calculate_model_size_accuracy,
    model_size_label,
    output_path_for_dataset,
    parse_model_family_size,
    plot_accuracy_by_model_size,
    save_figure,
)


def _result(
    *,
    dataset: str,
    model: str,
    pipeline: str,
    fmt: str | None,
    exact: float,
    rouge: float,
    bert: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model_id": model,
        "pipeline": pipeline,
        "format": fmt,
        "scores.normalized_exact_match": exact,
        "scores.rouge_l_f": rouge,
        "scores.bertscore_f": bert,
    }


def test_model_family_size_parsing_and_active_parameter_label() -> None:
    assert parse_model_family_size("Qwen3.5-0.8B-Q4_K_M.gguf") == ("Qwen", 0.8)
    assert parse_model_family_size("Ministral-3-14B-Instruct-Q4.gguf") == (
        "Ministral",
        14.0,
    )
    assert parse_model_family_size("gemma-4-E4B-it-Q4.gguf") == ("Gemma", 4.0)
    assert parse_model_family_size("gemma-4-26B-A4B-it-Q4.gguf") == ("Gemma", 26.0)
    assert model_size_label("gemma-4-26B-A4B-it-Q4.gguf") == "26\n(A4 active)"
    assert parse_model_family_size("unknown-model.gguf") is None


def test_model_size_accuracy_keeps_dataset_means_zeroes_and_sql() -> None:
    results = pd.DataFrame(
        [
            _result(
                dataset="bird",
                model="Qwen3.5-2B-Q4_K_M.gguf",
                pipeline="direct",
                fmt="csv",
                exact=0.0,
                rouge=0.0,
                bert=-0.1,
            ),
            _result(
                dataset="bird",
                model="Qwen3.5-2B-Q4_K_M.gguf",
                pipeline="direct",
                fmt="csv",
                exact=1.0,
                rouge=0.8,
                bert=0.5,
            ),
            _result(
                dataset="spider",
                model="Qwen3.5-2B-Q4_K_M.gguf",
                pipeline="text_to_sql",
                fmt=None,
                exact=0.7,
                rouge=0.8,
                bert=0.9,
            ),
            _result(
                dataset="bird",
                model="unsupported.gguf",
                pipeline="direct",
                fmt="csv",
                exact=1.0,
                rouge=1.0,
                bert=1.0,
            ),
        ]
    )

    summary = calculate_model_size_accuracy(results)
    indexed = summary.set_index(["dataset", "model_id", "method", "metric"])

    qwen_csv = indexed.loc[("bird", "Qwen3.5-2B-Q4_K_M.gguf", "csv")]
    assert qwen_csv.loc["normalized_exact_match", "mean_score"] == pytest.approx(0.5)
    assert qwen_csv.loc["rouge_l_f", "mean_score"] == pytest.approx(0.4)
    assert qwen_csv.loc["bertscore_f", "mean_score"] == pytest.approx(0.2)
    assert qwen_csv.loc["rouge_l_f", "sample_count"] == 2
    assert qwen_csv.loc["rouge_l_f", "model_family"] == "Qwen"
    assert qwen_csv.loc["rouge_l_f", "model_size_b"] == 2.0

    sql = indexed.loc[
        ("spider", "Qwen3.5-2B-Q4_K_M.gguf", "SQL execution", "rouge_l_f")
    ]
    assert sql["mean_score"] == pytest.approx(0.8)
    assert "unsupported.gguf" not in set(summary["model_id"])


def test_model_size_figure_has_metric_rows_family_columns_and_gaps(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    family_models = {
        "Gemma": (
            ("gemma-4-E2B-it-Q4.gguf", 2.0, "2"),
            ("gemma-4-E4B-it-Q4.gguf", 4.0, "4"),
        ),
        "Qwen": (
            ("Qwen3.5-2B-Q4.gguf", 2.0, "2"),
            ("Qwen3.5-4B-Q4.gguf", 4.0, "4"),
        ),
        "Ministral": (
            ("Ministral-3-8B-Q4.gguf", 8.0, "8"),
            ("Ministral-3-14B-Q4.gguf", 14.0, "14"),
        ),
    }
    metrics = ("normalized_exact_match", "rouge_l_f", "bertscore_f")
    for family, models in family_models.items():
        for model_index, (model, size, size_label) in enumerate(models):
            for method_index, method in enumerate(("csv", "SQL execution")):
                if family == "Gemma" and model_index == 0 and method == "SQL execution":
                    continue
                for metric_index, metric in enumerate(metrics):
                    rows.append(
                        {
                            "dataset": "spider",
                            "model_id": model,
                            "model_family": family,
                            "model_size_b": size,
                            "size_label": size_label,
                            "method": method,
                            "metric": metric,
                            "mean_score": 0.1 * (model_index + method_index + metric_index),
                            "sample_count": 2,
                            "datasets": ("spider",),
                            "rank": 1.0,
                        }
                    )
    summary = pd.DataFrame(rows)

    figure, axes = plot_accuracy_by_model_size(
        summary,
        dataset="spider",
        methods=("csv", "SQL execution"),
    )
    output = tmp_path / "nested" / "model-size.png"
    save_figure(figure, output)

    assert len(axes) == 3
    assert all(len(row) == 3 for row in axes)
    assert [axis.get_title() for axis in axes[0]] == ["Gemma", "Qwen", "Ministral"]
    assert [axis.get_ylabel() for axis in (row[0] for row in axes)] == [
        "Normalized Exact Match",
        "ROUGE-L F1",
        "BERTScore F1",
    ]
    assert all(axis.get_xscale() == "log" for row in axes for axis in row)
    assert any(pd.isna(value) for value in axes[0][0].lines[1].get_ydata())
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["CSV", "SQL"]
    assert output.is_file()
    assert output.stat().st_size > 0


def test_output_path_is_dataset_specific(tmp_path: Path) -> None:
    assert output_path_for_dataset(tmp_path, "bird") == (
        tmp_path / "accuracy_by_model_size_bird.png"
    )
    assert output_path_for_dataset(tmp_path, "spider") == (
        tmp_path / "accuracy_by_model_size_spider.png"
    )
