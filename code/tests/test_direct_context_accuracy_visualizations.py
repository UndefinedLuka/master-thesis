"""Tests for direct-context retrieval-accuracy calculations and figures."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd

from visualizations.config import (
    DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME,
    DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME,
)
from visualizations.direct_accuracy_by_format import (
    generate_direct_accuracy_by_format_figure,
    plot_direct_accuracy_by_format,
)
from visualizations.direct_accuracy_by_model_size import (
    generate_direct_accuracy_by_model_size_figure,
    plot_direct_accuracy_by_model_size,
)
from visualizations.direct_accuracy_data import (
    calculate_format_accuracy,
    calculate_model_size_accuracy,
    model_size_label,
    parse_model_family_size,
    validate_direct_accuracy_scores,
)


def _result(
    *,
    dataset: str = "spider",
    db_id: str = "db",
    model: str = "Qwen3.5-2B-Q4_K_M.gguf",
    pipeline: str = "direct",
    fmt: str | None = "csv",
    example_id: str = "spider:db:0",
    exact: float | None = 0.5,
    bert: float | None = 0.4,
    rouge: float | None = 0.6,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "db_id": db_id,
        "pipeline": pipeline,
        "format": fmt,
        "model_id": model,
        "example_id": example_id,
        "scores.normalized_exact_match": exact,
        "scores.bertscore_f": bert,
        "scores.rouge_l_f": rouge,
    }


def test_format_accuracy_is_direct_only_keeps_zeroes_and_excludes_qwen_27b() -> None:
    results = pd.DataFrame(
        [
            _result(exact=0.0, bert=-0.2, rouge=0.0),
            _result(example_id="spider:db:1", exact=1.0, bert=0.6, rouge=0.8),
            _result(
                pipeline="text_to_sql",
                fmt=None,
                example_id="spider:db:2",
                exact=1.0,
                bert=1.0,
                rouge=1.0,
            ),
            _result(
                model="Qwen3.5-27B-Q4_K_M.gguf",
                example_id="spider:db:3",
                exact=1.0,
                bert=1.0,
                rouge=1.0,
            ),
        ]
    )

    summary = calculate_format_accuracy(results).set_index("metric")

    assert summary.loc["normalized_exact_match", "mean_score"] == pytest.approx(0.5)
    assert summary.loc["bertscore_f", "mean_score"] == pytest.approx(0.2)
    assert summary.loc["rouge_l_f", "mean_score"] == pytest.approx(0.4)
    assert set(summary["model_count"]) == {1}
    assert set(summary["question_count"]) == {2}
    assert set(summary["observation_count"]) == {2}


def test_score_validation_reports_missing_direct_metric() -> None:
    results = pd.DataFrame([_result(bert=None)])

    with pytest.raises(ValueError, match="missing bertscore_f"):
        validate_direct_accuracy_scores(results)


@pytest.mark.parametrize("calculate", [calculate_format_accuracy, calculate_model_size_accuracy])
@pytest.mark.parametrize("invalid", [None, float("inf"), float("-inf")])
def test_calculations_reject_invalid_scores_without_changing_cohort(calculate, invalid) -> None:
    results = pd.DataFrame([_result(bert=invalid)])
    with pytest.raises(ValueError, match="missing bertscore_f|non-finite bertscore_f"):
        calculate(results)


@pytest.mark.parametrize("calculate", [calculate_format_accuracy, calculate_model_size_accuracy])
def test_calculations_reject_duplicate_runs(calculate) -> None:
    results = pd.DataFrame([_result(), _result()])
    with pytest.raises(ValueError, match="Duplicate direct-context"):
        calculate(results)


def test_answerable_failures_keep_run_weighting_pairing_and_model_effects() -> None:
    rows = []
    for db_id, question_count in (("small", 1), ("large", 3)):
        for question in range(question_count):
            for model in ("Qwen3.5-2B-Q4.gguf", "Qwen3.5-4B-Q4.gguf"):
                for fmt in ("csv", "json"):
                    failed = model.startswith("Qwen3.5-2B") or db_id == "large"
                    score = 0.0 if failed else 1.0
                    row = _result(
                        db_id=db_id, example_id=f"spider:{db_id}:{question}", model=model,
                        fmt=fmt, exact=score, bert=score, rouge=score,
                    )
                    row.update(error="context_exceeded" if failed else None,
                               has_answer=not failed)
                    rows.append(row)
    results = pd.DataFrame(rows)
    formats = calculate_format_accuracy(results)
    sizes = calculate_model_size_accuracy(results)
    assert formats["mean_score"].eq(0.125).all()
    assert formats["observation_count"].eq(8).all()
    assert formats["mean_score_effect"].eq(0).all()
    larger = sizes[sizes["model_size_b"].eq(4)]
    assert larger["mean_score"].eq(0.25).all()
    assert larger["mean_score_effect"].eq(0.25).all()
    assert larger["mean_score_effect_low"].eq(0).all()
    assert larger["mean_score_effect_high"].eq(1).all()
    assert sizes["observation_count"].eq(8).all()
    pd.testing.assert_frame_equal(sizes, calculate_model_size_accuracy(results))


def test_model_family_size_parsing_and_active_size_label() -> None:
    assert parse_model_family_size("Qwen3.5-0.8B-Q4.gguf") == ("qwen-3.5", 0.8)
    assert parse_model_family_size("Ministral-3-14B-Instruct-Q4.gguf") == (
        "ministral-3",
        14.0,
    )
    assert parse_model_family_size("gemma-4-E4B-it-Q4.gguf") == ("gemma-4", 4.0)
    assert parse_model_family_size("gemma-4-26B-A4B-it-Q4.gguf") == (
        "gemma-4",
        26.0,
    )
    assert model_size_label("gemma-4-26B-A4B-it-Q4.gguf") == "26 (A4 active)"
    assert parse_model_family_size("unsupported.gguf") is None


def test_model_size_accuracy_averages_all_formats_and_questions() -> None:
    results = pd.DataFrame(
        [
            _result(fmt="csv", exact=0.0, bert=0.1, rouge=0.2),
            _result(fmt="json", exact=1.0, bert=0.5, rouge=0.8),
            _result(
                fmt="csv",
                example_id="spider:db:1",
                exact=1.0,
                bert=0.3,
                rouge=0.6,
            ),
            _result(
                fmt="json",
                example_id="spider:db:1",
                exact=0.0,
                bert=0.7,
                rouge=0.4,
            ),
        ]
    )

    summary = calculate_model_size_accuracy(results).set_index("metric")

    assert summary.loc["normalized_exact_match", "mean_score"] == pytest.approx(0.5)
    assert summary.loc["bertscore_f", "mean_score"] == pytest.approx(0.4)
    assert summary.loc["rouge_l_f", "mean_score"] == pytest.approx(0.5)
    assert set(summary["format_count"]) == {2}
    assert set(summary["question_count"]) == {2}
    assert set(summary["observation_count"]) == {4}


def _plot_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = (
        "Qwen3.5-2B-Q4.gguf",
        "Qwen3.5-4B-Q4.gguf",
        "Ministral-3-8B-Q4.gguf",
        "Ministral-3-14B-Q4.gguf",
        "gemma-4-E2B-Q4.gguf",
        "gemma-4-E4B-Q4.gguf",
    )
    for dataset_index, dataset in enumerate(("spider", "bird")):
        for model_index, model in enumerate(models):
            for format_index, fmt in enumerate(("csv", "json")):
                score = 0.05 * (dataset_index + model_index + format_index)
                rows.append(
                    _result(
                        dataset=dataset,
                        model=model,
                        fmt=fmt,
                        example_id=f"{dataset}:db:0",
                        exact=score,
                        bert=score - 0.1,
                        rouge=score + 0.1,
                    )
                )
    return pd.DataFrame(rows)


def test_format_plot_preserves_nan_for_formats_missing_from_one_dataset(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        "visualizations.direct_accuracy_by_format.tight_layout", lambda *args, **kwargs: None,
    )
    summary = calculate_format_accuracy(_plot_results())
    summary = summary[~(summary["dataset"].eq("bird") & summary["format"].eq("json"))]

    figure, axes = plot_direct_accuracy_by_format(summary)
    try:
        for row in axes:
            assert len(row[0].patches) == len(row[1].patches) == 2
            assert not pd.isna(row[0].patches[1].get_height())
            assert pd.isna(row[1].patches[1].get_height())
    finally:
        plt.close(figure)


def test_format_and_model_size_figures_have_dataset_metric_grid(tmp_path: Path) -> None:
    results = _plot_results()
    format_figure, format_axes = plot_direct_accuracy_by_format(
        calculate_format_accuracy(results)
    )
    size_figure, size_axes = plot_direct_accuracy_by_model_size(
        calculate_model_size_accuracy(results)
    )

    assert len(format_axes) == 3
    assert all(len(row) == 2 for row in format_axes)
    assert [axis.get_title() for axis in format_axes[0]] == ["Spider", "BIRD"]
    assert [row[0].get_ylabel() for row in format_axes] == [
        "NEM\nMean score",
        "BERTScore\nMean score",
        "ROUGE-L\nMean score",
    ]
    assert len(size_axes) == 3
    assert all(len(row) == 2 for row in size_axes)
    assert all(axis.get_xscale() == "log" for row in size_axes for axis in row)
    for row in size_axes:
        for axis in row:
            assert all(
                tick.get_text().startswith(r"\num{") for tick in axis.get_xticklabels()
            )
            # Automatic locators may choose fractional steps below two decimal places.
            assert axis.yaxis.get_major_formatter()(0.025, 0) == r"\num{0.025}"
    assert [text.get_text() for text in size_figure.legends[0].get_texts()] == [
        "Qwen 3.5",
        "Ministral 3",
        "Gemma 4",
    ]

    figures_dir = tmp_path / "figures"
    format_path = generate_direct_accuracy_by_format_figure(results, figures_dir)
    size_path = generate_direct_accuracy_by_model_size_figure(results, figures_dir)
    assert format_path == figures_dir / DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME
    assert size_path == figures_dir / DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME
    assert format_path.stat().st_size > 0
    assert size_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(format_figure)
    plt.close(size_figure)
