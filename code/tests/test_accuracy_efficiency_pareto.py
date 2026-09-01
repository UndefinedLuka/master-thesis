"""Tests for accuracy-efficiency Pareto visualizations."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.accuracy_efficiency_pareto import (  # noqa: E402
    calculate_accuracy_efficiency_points,
    output_path_for_dataset,
    pareto_frontier,
    plot_accuracy_efficiency_pareto,
    save_figure,
)


def _result(
    *,
    dataset: str,
    model: str,
    pipeline: str,
    fmt: str | None,
    tokens: int,
    latency: float,
    exact: float,
    rouge: float,
    bert: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model_id": model,
        "pipeline": pipeline,
        "format": fmt,
        "input_tokens": tokens,
        "total_time_s": latency,
        "scores.normalized_exact_match": exact,
        "scores.rouge_l_f": rouge,
        "scores.bertscore_f": bert,
    }


def test_aggregation_includes_zero_accuracy_but_excludes_unmeasured_efficiency() -> None:
    results = pd.DataFrame(
        [
            _result(
                dataset="bird",
                model="Qwen3.5-2B-Q4_K_M.gguf",
                pipeline="direct",
                fmt="csv",
                tokens=100,
                latency=2.0,
                exact=1.0,
                rouge=0.8,
                bert=0.7,
            ),
            _result(
                dataset="bird",
                model="Qwen3.5-2B-Q4_K_M.gguf",
                pipeline="direct",
                fmt="csv",
                tokens=0,
                latency=0.0,
                exact=0.0,
                rouge=0.0,
                bert=0.0,
            ),
            _result(
                dataset="bird",
                model="gemma-4-E2B-it-Q4_K_M.gguf",
                pipeline="text_to_sql",
                fmt=None,
                tokens=500,
                latency=1.0,
                exact=0.6,
                rouge=0.7,
                bert=0.8,
            ),
        ]
    )

    points = calculate_accuracy_efficiency_points(results).set_index(["model_id", "method"])
    csv = points.loc[("Qwen3.5-2B-Q4_K_M.gguf", "csv")]
    sql = points.loc[("gemma-4-E2B-it-Q4_K_M.gguf", "SQL execution")]

    assert csv["normalized_exact_match"] == pytest.approx(0.5)
    assert csv["rouge_l_f"] == pytest.approx(0.4)
    assert csv["bertscore_f"] == pytest.approx(0.35)
    assert csv["accuracy_sample_count"] == 2
    assert csv["mean_input_tokens"] == 100
    assert csv["mean_total_time_s"] == 2.0
    assert csv["efficiency_sample_count"] == 1
    assert csv["model_family"] == "Qwen"
    assert sql["model_family"] == "Gemma"


def test_pareto_frontier_keeps_only_non_dominated_points() -> None:
    points = pd.DataFrame(
        {
            "name": ["cheap", "balanced", "dominated", "accurate", "duplicate"],
            "cost": [1.0, 2.0, 3.0, 4.0, 2.0],
            "score": [0.5, 0.7, 0.6, 0.9, 0.7],
        }
    )

    frontier = pareto_frontier(points, cost_column="cost", score_column="score")

    assert frontier["name"].tolist() == ["cheap", "balanced", "duplicate", "accurate"]


def test_pareto_figure_has_six_log_cost_panels_and_frontier_labels(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    model_specs = (
        ("Ministral-3-8B-Instruct-2512-Q4_K_M.gguf", "Ministral", 0.5),
        ("Qwen3.5-4B-Q4_K_M.gguf", "Qwen", 0.7),
        ("gemma-4-E4B-it-Q4_K_M.gguf", "Gemma", 0.9),
    )
    for model_index, (model, family, score) in enumerate(model_specs):
        for method_index, method in enumerate(("csv", "SQL execution")):
            rows.append(
                {
                    "dataset": "bird",
                    "model_id": model,
                    "model_family": family,
                    "method": method,
                    "mean_input_tokens": 100 * (model_index + 1) * (method_index + 1),
                    "mean_total_time_s": float((model_index + 1) * (method_index + 1)),
                    "normalized_exact_match": score,
                    "rouge_l_f": score - 0.1,
                    "bertscore_f": score + 0.05,
                    "accuracy_sample_count": 2,
                    "efficiency_sample_count": 2,
                }
            )
    points = pd.DataFrame(rows)

    figure, axes = plot_accuracy_efficiency_pareto(points, dataset="bird")
    output = tmp_path / "nested" / "pareto.png"
    save_figure(figure, output)

    flat_axes = [axis for row in axes for axis in row]
    assert len(flat_axes) == 6
    assert all(axis.get_xscale() == "log" for axis in flat_axes)
    assert [axis.get_title() for axis in axes[0]] == [
        "Normalized Exact Match",
        "ROUGE-L F1",
        "BERTScore F1",
    ]
    assert [axis.get_xlabel() for axis in axes[0]] == ["Mean input tokens"] * 3
    assert [axis.get_xlabel() for axis in axes[1]] == ["Mean total latency (seconds)"] * 3
    legend_labels = [
        text.get_text() for legend in figure.legends for text in legend.get_texts()
    ]
    assert {"CSV", "SQL", "Ministral", "Qwen", "Gemma", "Pareto frontier"}.issubset(
        legend_labels
    )
    assert any(axis.texts for axis in flat_axes)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_output_path_is_dataset_specific(tmp_path: Path) -> None:
    assert output_path_for_dataset(tmp_path, "bird") == (
        tmp_path / "accuracy_efficiency_pareto_bird.png"
    )
    assert output_path_for_dataset(tmp_path, "spider") == (
        tmp_path / "accuracy_efficiency_pareto_spider.png"
    )
