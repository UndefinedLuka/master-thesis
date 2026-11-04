"""Regression coverage for answerable-cohort cross-pipeline accuracy and pairing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from visualizations.config import DIRECT_ACCURACY_METRIC_ORDER, SERIALIZATION_FORMAT_ORDER
from visualizations.pipeline_retrieval_comparison import (
    calculate_pipeline_retrieval_comparison,
    generate_pipeline_retrieval_comparison_figure,
    plot_pipeline_retrieval_comparison,
)
from visualizations.uncertainty import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED


def _matrix() -> pd.DataFrame:
    rows = []
    for dataset in ("spider", "bird"):
        for db, questions in (("small", 1), ("large", 3)):
            for question in range(questions):
                for model in ("Qwen3.5-2B", "gemma-4-E4B"):
                    for method in (*SERIALIZATION_FORMAT_ORDER, "text_to_sql"):
                        sql = method == "text_to_sql"
                        nem = int((db == "small") != sql)
                        rows.append({
                            "dataset": dataset, "db_id": db, "example_id": f"{db}:{question}",
                            "model_id": model, "pipeline": "text_to_sql" if sql else "direct",
                            "format": None if sql else method, "error": None,
                            "execution_success": True if sql else None, "has_answer": True,
                            "scores.normalized_exact_match": nem,
                            "scores.bertscore_f": 0.2 + 0.6 * nem,
                            "scores.rouge_l_f": 0.1 + 0.8 * nem,
                        })
    return pd.DataFrame(rows)


def test_run_weighted_means_and_paired_database_bootstrap_all_metrics() -> None:
    data = _matrix()
    analysis = calculate_pipeline_retrieval_comparison(data.sample(frac=1, random_state=7))
    assert len(analysis.records) == len(data)
    assert len(analysis.records.query("method == 'text_to_sql'")) == 16
    assert len(analysis.summary) == 2 * 7 * 3
    # Independent bootstrap oracle: databases sorted large, small with 6 versus 2 runs.
    assert BOOTSTRAP_DRAWS == 10_000
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(2, [0.5, 0.5], size=10_000)
    for metric, (base, span) in zip(DIRECT_ACCURACY_METRIC_ORDER,
                                    ((0, 1), (0.2, 0.6), (0.1, 0.8)), strict=True):
        direct_draws = weights @ np.array([6 * base, 2 * (base + span)]) / (weights @ [6, 2])
        sql_draws = weights @ np.array([6 * (base + span), 2 * base]) / (weights @ [6, 2])
        for dataset in ("spider", "bird"):
            panel = analysis.summary.loc[
                analysis.summary["dataset"].eq(dataset) & analysis.summary["metric"].eq(metric)
            ]
            sql_row = panel.set_index("method").loc["text_to_sql"]
            assert sql_row["observation_count"] == 8
            assert sql_row["mean_score"] == pytest.approx(base + 0.75 * span)
            assert sql_row["mean_score_effect"] == 0
            for row in panel.query("method != 'text_to_sql'").itertuples():
                assert row.observation_count == 8
                assert row.mean_score == pytest.approx(base + 0.25 * span)
                assert row.mean_score_effect == pytest.approx(-0.5 * span)
                assert (row.mean_score_low, row.mean_score_high) == pytest.approx(
                    np.quantile(direct_draws, [0.025, 0.975]))
                assert (row.mean_score_effect_low, row.mean_score_effect_high) == pytest.approx(
                    np.quantile(direct_draws - sql_draws, [0.025, 0.975]))


@pytest.mark.parametrize("failure", ["context", "sql_execution", "sql_extraction"])
def test_failures_are_retained_and_require_stored_zeroes(failure: str) -> None:
    data = _matrix()
    pipeline = "direct" if failure == "context" else "text_to_sql"
    index = data.index[data["pipeline"].eq(pipeline)][0]
    if failure == "context":
        data.loc[index, "error"] = "context_exceeded"
    elif failure == "sql_execution":
        data.loc[index, "execution_success"] = False
    else:
        data.loc[index, "error"] = "no_sql_extracted"
    data.loc[index, "has_answer"] = False
    with pytest.raises(ValueError, match="Failed outputs must have zero"):
        calculate_pipeline_retrieval_comparison(data)
    data.loc[index, [f"scores.{metric}" for metric in DIRECT_ACCURACY_METRIC_ORDER]] = 0
    analysis = calculate_pipeline_retrieval_comparison(data)
    assert len(analysis.records) == len(data)
    assert (analysis.summary["observation_count"] == 8).all()


def test_blank_outputs_do_not_select_the_cohort_and_negative_bertscore_is_valid() -> None:
    data = _matrix()
    data.loc[0, "has_answer"] = False
    data.loc[0, "scores.bertscore_f"] = -0.1
    analysis = calculate_pipeline_retrieval_comparison(data)
    assert len(analysis.records) == len(data)
    assert analysis.records.loc[0, "scores.bertscore_f"] == -0.1


@pytest.mark.parametrize("pipeline", ["direct", "text_to_sql"])
def test_actual_bertscore_roundoff_is_preserved_in_records_and_estimates(pipeline: str) -> None:
    data = _matrix()
    observed_maximum = 1.0000011870480279
    selected = data["pipeline"].eq(pipeline)
    data.loc[selected, "scores.bertscore_f"] = observed_maximum
    original = data.copy(deep=True)
    analysis = calculate_pipeline_retrieval_comparison(data)
    pd.testing.assert_frame_equal(data, original)
    assert analysis.records.loc[selected, "scores.bertscore_f"].eq(observed_maximum).all()
    summary = analysis.summary.loc[analysis.summary["metric"].eq("bertscore_f")]
    selected_methods = summary["method"].eq("text_to_sql")
    if pipeline == "direct":
        selected_methods = ~selected_methods
    assert summary.loc[selected_methods, "mean_score"].eq(observed_maximum).all()
    expected_effect = observed_maximum - 0.65 if pipeline == "direct" else 0.35 - observed_maximum
    np.testing.assert_allclose(
        summary.loc[summary["method"].ne("text_to_sql"), "mean_score_effect"],
        expected_effect, rtol=0, atol=1e-15,
    )


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_database", "wrong_model",
                                     "missing_format", "null_identity"])
def test_incomplete_or_corrupted_pairing_fails_instead_of_changing_cohort(problem: str) -> None:
    data = _matrix()
    if problem == "missing":
        data = data.drop(0)
    elif problem == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif problem == "wrong_database":
        data.loc[0, "db_id"] = "different"
    elif problem == "wrong_model":
        data.loc[0, "model_id"] = "different"
    elif problem == "missing_format":
        data = data.loc[data["format"].ne("csv")]
    else:
        data.loc[0, "example_id"] = None
    with pytest.raises(ValueError):
        calculate_pipeline_retrieval_comparison(data)


@pytest.mark.parametrize("metric,value", [
    ("normalized_exact_match", 0.5), ("rouge_l_f", -0.1), ("rouge_l_f", 1.1),
    ("bertscore_f", -1.1), ("bertscore_f", 1.1), ("bertscore_f", np.inf),
    ("bertscore_f", 1.0001), ("bertscore_f", -1.0001),
    ("rouge_l_f", np.nan), ("normalized_exact_match", "invalid"),
])
def test_invalid_scores_are_rejected(metric: str, value: object) -> None:
    data = _matrix()
    data[f"scores.{metric}"] = data[f"scores.{metric}"].astype(object)
    data.loc[0, f"scores.{metric}"] = value
    with pytest.raises(ValueError, match="Invalid"):
        calculate_pipeline_retrieval_comparison(data)


def test_excluded_models_are_removed_before_validation_and_input_is_not_modified() -> None:
    data = _matrix()
    excluded = data.iloc[[0]].copy()
    excluded["model_id"] = "Qwen3.5-27B-Q4"
    excluded["scores.bertscore_f"] = np.nan
    data = pd.concat([data, excluded], ignore_index=True)
    original = data.copy(deep=True)
    analysis = calculate_pipeline_retrieval_comparison(data)
    assert len(analysis.records) == len(data) - 1
    pd.testing.assert_frame_equal(original, data)
    with pytest.raises(ValueError):
        calculate_pipeline_retrieval_comparison(data, excluded_model_markers=())


def test_one_database_has_no_spurious_interval() -> None:
    data = _matrix().query("db_id == 'small'")
    analysis = calculate_pipeline_retrieval_comparison(data)
    assert analysis.summary["mean_score_effect_low"].isna().all()
    assert analysis.summary["mean_score_high"].isna().all()


def test_display_covers_both_datasets_all_metrics_and_formats(tmp_path) -> None:
    import matplotlib.pyplot as plt

    analysis = calculate_pipeline_retrieval_comparison(_matrix())
    figure = plot_pipeline_retrieval_comparison(analysis)
    assert len(figure.axes) == 6
    for index, axis in enumerate(figure.axes):
        assert len(axis.get_yticklabels()) == 7 or not axis.get_yticklabels()
        assert len(axis.collections) == 7  # absolute mean interval for SQL and each format
        points = [line for line in axis.lines if line.get_marker() == "o"]
        dataset = ("spider", "bird")[index % 2]
        metric = DIRECT_ACCURACY_METRIC_ORDER[index // 2]
        panel = analysis.summary.loc[
            analysis.summary["dataset"].eq(dataset) & analysis.summary["metric"].eq(metric)
        ]
        np.testing.assert_allclose(
            [line.get_xdata()[0] for line in points], panel["mean_score"],
        )
        for collection, row in zip(axis.collections, panel.itertuples(), strict=True):
            np.testing.assert_allclose(
                collection.get_segments()[0][:, 0], [row.mean_score_low, row.mean_score_high],
            )
    plt.close(figure)
    path = generate_pipeline_retrieval_comparison_figure(analysis, tmp_path)
    assert path.name == "pipeline_retrieval_comparison.pdf"
    assert path.read_bytes().startswith(b"%PDF")
