"""Tests for the mean-accuracy LaTeX table generator."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")

import pandas as pd

from visualizations.accuracy_table import (
    add_per_column_ranks,
    calculate_accuracy_means,
    output_path_for_dataset,
    render_accuracy_table,
    render_dataset_accuracy_tables,
    write_latex_table,
)


def _result(
    *,
    model: str,
    pipeline: str,
    fmt: str | None,
    dataset: str,
    rouge: float,
    bert: float | None,
    exact: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "format": fmt,
        "model_id": model,
        "scores.rouge_l_f": rouge,
        "scores.bertscore_f": bert,
        "scores.normalized_exact_match": exact,
    }


def test_calculate_accuracy_means_includes_zeroes_and_maps_sql() -> None:
    results = pd.DataFrame(
        [
            _result(
                model="model-a.gguf",
                pipeline="direct",
                fmt="csv",
                dataset="bird",
                rouge=0.0,
                bert=0.2,
                exact=0.0,
            ),
            _result(
                model="model-a.gguf",
                pipeline="direct",
                fmt="csv",
                dataset="spider",
                rouge=1.0,
                bert=0.4,
                exact=1.0,
            ),
            _result(
                model="model-a.gguf",
                pipeline="text_to_sql",
                fmt=None,
                dataset="bird",
                rouge=0.8,
                bert=0.9,
                exact=0.7,
            ),
        ]
    )

    summary = calculate_accuracy_means(results).set_index(["method", "metric"])

    assert summary.loc[("csv", "rouge_l_f"), "mean_score"] == pytest.approx(0.5)
    assert summary.loc[("csv", "bertscore_f"), "mean_score"] == pytest.approx(0.3)
    assert summary.loc[("csv", "normalized_exact_match"), "mean_score"] == pytest.approx(0.5)
    assert summary.loc[("csv", "rouge_l_f"), "sample_count"] == 2
    assert summary.loc[("csv", "rouge_l_f"), "datasets"] == ("bird", "spider")
    assert summary.loc[("SQL execution", "rouge_l_f"), "mean_score"] == pytest.approx(0.8)
    assert summary.loc[("SQL execution", "rouge_l_f"), "datasets"] == ("bird",)


def test_dataset_tables_use_independent_means_rankings_labels_and_paths(tmp_path: Path) -> None:
    results = pd.DataFrame(
        [
            _result(
                model=model,
                pipeline="direct",
                fmt="csv",
                dataset=dataset,
                rouge=score,
                bert=score,
                exact=score,
            )
            for dataset, model, score in (
                ("bird", "model-a.gguf", 0.9),
                ("bird", "model-b.gguf", 0.1),
                ("spider", "model-a.gguf", 0.2),
                ("spider", "model-b.gguf", 0.8),
            )
        ]
    )

    tables = render_dataset_accuracy_tables(results, datasets=("bird", "spider"))

    assert list(tables) == ["bird", "spider"]
    assert r"\label{tab:mean-accuracy-by-model-method-bird}" in tables["bird"]
    assert r"\label{tab:mean-accuracy-by-model-method-spider}" in tables["spider"]
    assert "Results are reported for BIRD." in tables["bird"]
    assert "Results are reported for Spider." in tables["spider"]
    assert r"model-a & \textbf{0.900}" in tables["bird"]
    assert r"model-b & \textbf{0.800}" in tables["spider"]

    base_path = tmp_path / "mean_accuracy_by_model_and_format.tex"
    assert output_path_for_dataset(base_path, "bird").name == (
        "mean_accuracy_by_model_and_format_bird.tex"
    )
    assert output_path_for_dataset(base_path, "spider").name == (
        "mean_accuracy_by_model_and_format_spider.tex"
    )


def test_ranking_is_per_method_metric_and_uses_unrounded_means() -> None:
    summary = pd.DataFrame(
        {
            "model_id": ["a", "b", "c", "a", "b", "c"],
            "method": ["csv", "csv", "csv", "json", "json", "json"],
            "metric": ["rouge_l_f"] * 6,
            "mean_score": [0.5004, 0.5005, 0.1, 0.2, 0.9, 0.8],
        }
    )

    ranked = add_per_column_ranks(summary).set_index(["method", "model_id"])

    assert ranked.loc[("csv", "b"), "rank"] == 1
    assert ranked.loc[("csv", "a"), "rank"] == 2
    assert ranked.loc[("json", "b"), "rank"] == 1
    assert ranked.loc[("json", "c"), "rank"] == 2


def test_render_accuracy_table_groups_columns_and_highlights_top_two(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for model, score in (("model_a.gguf", 0.6), ("model_b.gguf", 0.9), ("model_c.gguf", 0.8)):
        for method in ("csv", "SQL execution"):
            for metric in ("rouge_l_f", "bertscore_f", "normalized_exact_match"):
                rows.append(
                    {
                        "model_id": model,
                        "method": method,
                        "metric": metric,
                        "mean_score": score,
                        "sample_count": 2,
                        "datasets": ("bird", "spider"),
                    }
                )
    summary = pd.DataFrame(rows)

    latex = render_accuracy_table(
        summary,
        methods=("csv", "SQL execution"),
        precision=3,
        caption="Accuracy & coverage",
        label="tab:test_accuracy",
    )
    output = tmp_path / "nested" / "accuracy.tex"
    write_latex_table(output, latex)

    assert r"\begin{landscape}" in latex
    assert r"\begin{tabular}{@{}l*{6}{r}@{}}" in latex
    assert r"\multicolumn{3}{c}{CSV}" in latex
    assert r"\multicolumn{3}{c}{SQL}" in latex
    assert latex.count("R-L & BERT-F & EM") == 2
    assert r"\caption{Accuracy \& coverage}" in latex
    assert r"model\_b" in latex
    assert latex.count(r"\textbf{0.900}") == 6
    assert latex.count(r"\underline{0.800}") == 6
    assert latex.count(r"0.600") == 6
    assert output.read_text(encoding="utf-8") == latex


def test_render_accuracy_table_marks_missing_cells() -> None:
    summary = pd.DataFrame(
        {
            "model_id": ["only-model.gguf"],
            "method": ["csv"],
            "metric": ["rouge_l_f"],
            "mean_score": [0.5],
            "sample_count": [1],
            "datasets": [("bird",)],
        }
    )

    latex = render_accuracy_table(
        summary,
        methods=("csv",),
        metrics=("rouge_l_f", "bertscore_f", "normalized_exact_match"),
    )

    assert r"only-model & \textbf{0.500} & -- & -- \\" in latex
