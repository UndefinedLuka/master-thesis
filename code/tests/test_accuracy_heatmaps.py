"""Tests for per-dataset retrieval-accuracy heatmaps."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.accuracy_heatmaps import (  # noqa: E402
    SCORE_MAX,
    SCORE_MIN,
    calculate_dataset_accuracy_means,
    output_path_for_dataset,
    plot_accuracy_heatmaps,
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


def test_dataset_means_are_separate_include_zeroes_and_map_sql() -> None:
    results = pd.DataFrame(
        [
            _result(
                dataset="bird",
                model="model-a.gguf",
                pipeline="direct",
                fmt="csv",
                exact=0.0,
                rouge=0.0,
                bert=-0.2,
            ),
            _result(
                dataset="bird",
                model="model-a.gguf",
                pipeline="direct",
                fmt="csv",
                exact=1.0,
                rouge=0.8,
                bert=0.4,
            ),
            _result(
                dataset="spider",
                model="model-a.gguf",
                pipeline="direct",
                fmt="csv",
                exact=1.0,
                rouge=1.0,
                bert=0.9,
            ),
            _result(
                dataset="bird",
                model="model-a.gguf",
                pipeline="text_to_sql",
                fmt=None,
                exact=0.7,
                rouge=0.8,
                bert=0.9,
            ),
        ]
    )

    summary = calculate_dataset_accuracy_means(results).set_index(
        ["dataset", "method", "metric"]
    )

    assert summary.loc[("bird", "csv", "normalized_exact_match"), "mean_score"] == 0.5
    assert summary.loc[("bird", "csv", "rouge_l_f"), "mean_score"] == pytest.approx(0.4)
    assert summary.loc[("bird", "csv", "bertscore_f"), "mean_score"] == pytest.approx(0.1)
    assert summary.loc[("spider", "csv", "normalized_exact_match"), "mean_score"] == 1.0
    assert summary.loc[("bird", "SQL execution", "rouge_l_f"), "mean_score"] == 0.8
    assert summary.loc[("bird", "csv", "rouge_l_f"), "sample_count"] == 2


def test_accuracy_heatmaps_show_three_metrics_and_missing_sql(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    models = ("Qwen3.5-2B-Q4_K_M.gguf", "gemma-4-E4B-it-Q4_K_M.gguf")
    methods = ("csv", "xml", "SQL execution")
    metrics = ("normalized_exact_match", "rouge_l_f", "bertscore_f")
    for model_index, model in enumerate(models):
        for method_index, method in enumerate(methods):
            if model_index == 0 and method == "SQL execution":
                continue
            for metric_index, metric in enumerate(metrics):
                rows.append(
                    {
                        "dataset": "spider",
                        "model_id": model,
                        "method": method,
                        "metric": metric,
                        "mean_score": (model_index + method_index + metric_index) / 10,
                        "sample_count": 2,
                        "datasets": ("spider",),
                        "rank": 1.0,
                    }
                )
    summary = pd.DataFrame(rows)

    figure, axes = plot_accuracy_heatmaps(
        summary,
        dataset="spider",
        methods=methods,
        metrics=metrics,
    )
    output = tmp_path / "nested" / "accuracy-heatmaps.png"
    save_figure(figure, output)

    assert [axis.get_title() for axis in axes] == [
        "Normalized Exact Match",
        "ROUGE-L F1",
        "BERTScore F1",
    ]
    assert [label.get_text() for label in axes[0].get_xticklabels()] == [
        "CSV",
        "XML",
        "SQL",
    ]
    assert [label.get_text() for label in axes[0].get_yticklabels()] == [
        "Qwen 3.5 2B",
        "Gemma 4 E4B",
    ]
    assert all(any(text.get_text() == "N/A" for text in axis.texts) for axis in axes)
    assert all(axis.collections[0].norm.vmin == SCORE_MIN for axis in axes)
    assert all(axis.collections[0].norm.vmax == SCORE_MAX for axis in axes)
    assert len(figure.axes) == 4  # three heatmaps plus one shared colorbar
    assert output.is_file()
    assert output.stat().st_size > 0


def test_output_path_is_dataset_specific(tmp_path: Path) -> None:
    assert output_path_for_dataset(tmp_path, "bird") == tmp_path / "accuracy_heatmaps_bird.png"
    assert output_path_for_dataset(tmp_path, "spider") == (
        tmp_path / "accuracy_heatmaps_spider.png"
    )
