"""Tests for paired format win-rate matrices."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.format_win_rate_matrix import (  # noqa: E402
    calculate_format_win_rates,
    output_path_for_dataset,
    plot_format_win_rate_matrices,
    save_figure,
)


def _result(
    *,
    dataset: str,
    model: str,
    example: str,
    pipeline: str,
    fmt: str | None,
    exact: float,
    rouge: float,
    bert: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model_id": model,
        "example_id": example,
        "pipeline": pipeline,
        "format": fmt,
        "scores.normalized_exact_match": exact,
        "scores.rouge_l_f": rouge,
        "scores.bertscore_f": bert,
    }


def test_win_rates_are_paired_by_model_question_and_keep_ties_in_denominator() -> None:
    results = pd.DataFrame(
        [
            _result(
                dataset="bird",
                model="model-a",
                example="q1",
                pipeline="direct",
                fmt="csv",
                exact=1.0,
                rouge=0.8,
                bert=0.7,
            ),
            _result(
                dataset="bird",
                model="model-a",
                example="q1",
                pipeline="direct",
                fmt="json",
                exact=0.0,
                rouge=0.6,
                bert=0.7,
            ),
            _result(
                dataset="bird",
                model="model-a",
                example="q2",
                pipeline="direct",
                fmt="csv",
                exact=0.0,
                rouge=0.2,
                bert=0.4,
            ),
            _result(
                dataset="bird",
                model="model-a",
                example="q2",
                pipeline="direct",
                fmt="json",
                exact=1.0,
                rouge=0.4,
                bert=0.3,
            ),
            _result(
                dataset="bird",
                model="model-b",
                example="q1",
                pipeline="direct",
                fmt="csv",
                exact=1.0,
                rouge=0.5,
                bert=0.5,
            ),
            _result(
                dataset="bird",
                model="model-b",
                example="q1",
                pipeline="direct",
                fmt="json",
                exact=1.0,
                rouge=0.5,
                bert=0.4,
            ),
            _result(
                dataset="bird",
                model="model-a",
                example="q1",
                pipeline="text_to_sql",
                fmt=None,
                exact=1.0,
                rouge=0.9,
                bert=0.9,
            ),
        ]
    )

    summary = calculate_format_win_rates(
        results,
        datasets=("bird",),
        methods=("csv", "json", "SQL execution", "xml"),
        metrics=("normalized_exact_match", "rouge_l_f", "bertscore_f"),
    ).set_index(["metric", "row_method", "column_method"])

    csv_json_exact = summary.loc[("normalized_exact_match", "csv", "json")]
    json_csv_exact = summary.loc[("normalized_exact_match", "json", "csv")]
    assert csv_json_exact["pair_count"] == 3
    assert csv_json_exact["win_count"] == 1
    assert csv_json_exact["loss_count"] == 1
    assert csv_json_exact["tie_count"] == 1
    assert csv_json_exact["win_rate_pct"] == pytest.approx(100 / 3)
    assert json_csv_exact["win_rate_pct"] == pytest.approx(100 / 3)

    csv_json_rouge = summary.loc[("rouge_l_f", "csv", "json")]
    assert csv_json_rouge["win_count"] == 1
    assert csv_json_rouge["loss_count"] == 1
    assert csv_json_rouge["tie_count"] == 1

    csv_json_bert = summary.loc[("bertscore_f", "csv", "json")]
    assert csv_json_bert["win_count"] == 2
    assert csv_json_bert["loss_count"] == 0
    assert csv_json_bert["tie_count"] == 1
    assert csv_json_bert["win_rate_pct"] == pytest.approx(200 / 3)

    # SQL overlaps only model-a/q1; model-b/q1 must not be cross-paired with it.
    sql_csv = summary.loc[("rouge_l_f", "SQL execution", "csv")]
    assert sql_csv["pair_count"] == 1
    assert sql_csv["win_rate_pct"] == 100.0

    missing = summary.loc[("rouge_l_f", "xml", "csv")]
    assert missing["pair_count"] == 0
    assert pd.isna(missing["win_rate_pct"])


def test_duplicate_model_question_method_rows_are_rejected() -> None:
    row = _result(
        dataset="bird",
        model="model-a",
        example="q1",
        pipeline="direct",
        fmt="csv",
        exact=1.0,
        rouge=1.0,
        bert=1.0,
    )

    with pytest.raises(ValueError, match="duplicate"):
        calculate_format_win_rates(
            pd.DataFrame([row, row]),
            datasets=("bird",),
            methods=("csv",),
            metrics=("rouge_l_f",),
        )


def test_win_rate_figure_has_three_matrices_annotations_and_masked_diagonal(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    methods = ("csv", "json", "SQL execution")
    metrics = ("normalized_exact_match", "rouge_l_f", "bertscore_f")
    for metric in metrics:
        for row_index, row_method in enumerate(methods):
            for column_index, column_method in enumerate(methods):
                pair_count = 0 if "SQL execution" in (row_method, column_method) else 100
                if row_method == column_method:
                    pair_count = 100
                rows.append(
                    {
                        "dataset": "spider",
                        "metric": metric,
                        "row_method": row_method,
                        "column_method": column_method,
                        "pair_count": pair_count,
                        "win_count": 30,
                        "loss_count": 50,
                        "tie_count": 20,
                        "win_rate_pct": (
                            float("nan")
                            if pair_count == 0
                            else float((row_index + column_index) * 20)
                        ),
                    }
                )
    summary = pd.DataFrame(rows)

    figure, axes = plot_format_win_rate_matrices(
        summary,
        dataset="spider",
        methods=methods,
        metrics=metrics,
    )
    output = tmp_path / "nested" / "win-rates.png"
    save_figure(figure, output)

    assert [axis.get_title() for axis in axes] == [
        "Normalized Exact Match",
        "ROUGE-L F1",
        "BERTScore F1",
    ]
    assert [label.get_text() for label in axes[0].get_xticklabels()] == [
        "CSV",
        "JSON",
        "SQL",
    ]
    assert all(sum(text.get_text() == "—" for text in axis.texts) == 3 for axis in axes)
    assert all(any(text.get_text() == "N/A" for text in axis.texts) for axis in axes)
    assert all(
        any("%\nn=" in text.get_text() for text in axis.texts)
        for axis in axes
    )
    assert len(figure.axes) == 4  # three matrices plus shared colorbar
    assert output.is_file()
    assert output.stat().st_size > 0


def test_output_path_is_dataset_specific(tmp_path: Path) -> None:
    assert output_path_for_dataset(tmp_path, "bird") == (
        tmp_path / "format_win_rate_matrix_bird.png"
    )
    assert output_path_for_dataset(tmp_path, "spider") == (
        tmp_path / "format_win_rate_matrix_spider.png"
    )
