"""Tests for answerable database-size associations and the retained NEM figure."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd

from visualizations.config import (
    DIRECT_DATABASE_SIZE_FIGURE_FILENAMES,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.direct_accuracy_by_database_size import (
    calculate_database_size_accuracy,
    generate_database_size_accuracy_figures,
    plot_database_size_accuracy,
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
        )


def test_nem_plot_has_six_format_panels_and_both_datasets(tmp_path: Path) -> None:
    results, report = _inputs()
    analysis = calculate_database_size_accuracy(
        results,
        report,
    )

    figure, axes = plot_database_size_accuracy(analysis)
    assert len(axes) == 6
    assert [axis.get_title() for axis in axes] == [
        "CSV", "TOON", "Markdown", "YAML", "JSON", "XML",
    ]
    assert all(axis.get_xscale() == "log" for axis in axes)
    assert all(len(axis.collections) == 2 for axis in axes)
    annotations = [text.get_text() for axis in axes for text in axis.texts]
    assert any("Interval unavailable" in text for text in annotations)
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["Spider", "BIRD"]

    outputs = generate_database_size_accuracy_figures(
        results,
        report,
        tmp_path,
        excluded_model_markers=(),
    )
    assert outputs == [
        tmp_path / DIRECT_DATABASE_SIZE_FIGURE_FILENAMES["normalized_exact_match"]
    ]
    assert all(path.stat().st_size > 0 for path in outputs)
    assert list(tmp_path.glob("*.pdf")) == outputs

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_coverage_uses_answerable_databases_not_full_serialization_inventory() -> None:
    results, report = _inputs()
    results = results[results["db_id"].ne("large_bird")]
    analysis = calculate_database_size_accuracy(results, report)
    counts = analysis.correlations.groupby("dataset")["database_count"].unique()
    assert list(counts["bird"]) == [1]
    assert list(counts["spider"]) == [2]


def test_coverage_rejects_missing_answerable_database_in_a_format() -> None:
    results, report = _inputs()
    results = results[~(results["db_id"].eq("large_bird") & results["format"].eq("xml"))]
    with pytest.raises(ValueError, match="Answerable accuracy database coverage.*bird/xml"):
        calculate_database_size_accuracy(results, report)


def test_associations_include_zero_scored_context_failures() -> None:
    results, report = _inputs()
    failed = results["db_id"].eq("large_bird")
    results.loc[failed, "error"] = "context_exceeded"
    results.loc[failed, "has_answer"] = False
    for metric in ("normalized_exact_match", "bertscore_f", "rouge_l_f"):
        results.loc[failed, f"scores.{metric}"] = 0.0
    analysis = calculate_database_size_accuracy(results, report)
    database = analysis.database_scores[analysis.database_scores["db_id"].eq("large_bird")]
    assert len(database) == 6 * 3
    assert database["mean_score"].eq(0).all()
    assert database["observation_count"].eq(2).all()
    assert analysis.correlations["database_count"].eq(2).all()


def test_all_metric_associations_have_reproducible_database_intervals() -> None:
    results, report = _inputs()
    # Four distinct database points allow >=95% non-degenerate bootstrap draws.
    extra_results = results.copy()
    extra_report = report.copy()
    extra_results["db_id"] += "_extra"
    extra_results["example_id"] += "_extra"
    extra_report["database_id"] += "_extra"
    extra_report["tokens"] *= 2
    for metric in ("normalized_exact_match", "bertscore_f", "rouge_l_f"):
        extra_results[f"scores.{metric}"] += 0.05
    results = pd.concat([results, extra_results], ignore_index=True)
    report = pd.concat([report, extra_report], ignore_index=True)
    first = calculate_database_size_accuracy(results, report).correlations
    second = calculate_database_size_accuracy(results, report).correlations
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {"normalized_exact_match", "bertscore_f", "rouge_l_f"}
    assert first[["pearson_log_size_low", "pearson_log_size_high"]].notna().all().all()
    assert (first["pearson_log_size_low"] <= first["pearson_log_size_high"]).all()
