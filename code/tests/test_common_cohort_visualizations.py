"""Tests for the strict cross-pipeline question intersection and its artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd

from visualizations.common_cohort_data import calculate_common_cohort_analysis
from visualizations.common_cohort_latency import (
    generate_common_cohort_latency_figure,
    plot_common_cohort_latency,
)
from visualizations.common_cohort_latency_by_model import (
    generate_common_cohort_model_figure,
    plot_common_cohort_latency_by_model,
)
from visualizations.common_cohort_operational_table import (
    generate_common_cohort_table,
    render_common_cohort_table,
)
from visualizations.common_cohort_token_usage import (
    generate_common_cohort_token_figure,
    plot_common_cohort_token_usage,
)
from visualizations.config import (
    COMMON_COHORT_LATENCY_FIGURE_FILENAME,
    COMMON_COHORT_MODEL_FIGURE_FILENAME,
    COMMON_COHORT_TABLE_FILENAME,
    COMMON_COHORT_TOKENS_FIGURE_FILENAME,
)
from visualizations.generate_all import generate_all
from visualizations.text_to_sql_data import ATTEMPT_GENERATION_COLUMN, ATTEMPT_TTFT_COLUMN


def _matrix() -> pd.DataFrame:
    rows = []
    for dataset in ("spider", "bird"):
        for question in (0, 1):
            for model in ("Qwen3.5-2B-Q4.gguf", "gemma-4-E4B-Q4.gguf"):
                for method in ("csv", "json", "text_to_sql"):
                    sql = method == "text_to_sql"
                    total = {"csv": 2.0, "json": 8.0, "text_to_sql": 1.1}[method]
                    rows.append(
                        {
                            "dataset": dataset,
                            "db_id": f"{dataset}-db",
                            "example_id": f"{dataset}:db:{question}",
                            "model_id": model,
                            "pipeline": "text_to_sql" if sql else "direct",
                            "format": None if sql else method,
                            "error": None,
                            "has_answer": True,
                            "input_tokens": {"csv": 100, "json": 400, "text_to_sql": 20}[method],
                            "output_tokens": 10,
                            "time_to_first_token_s": 0.2 if sql else total / 2,
                            "generation_time_s": 1.0 if sql else total,
                            "query_execution_time_s": 0.1 if sql else 0,
                            "total_time_s": total,
                            "retry_count": 0,
                            "execution_success": True if sql else None,
                            "scores.normalized_exact_match": 0,
                            "scores.bertscore_f": 0.4,
                            "scores.rouge_l_f": 0.5,
                            "execution.result_match": False if sql else None,
                            ATTEMPT_TTFT_COLUMN: [0.2] if sql else None,
                            ATTEMPT_GENERATION_COLUMN: [1.0] if sql else None,
                            "predicted_sql": "SELECT 1" if sql else None,
                            "definitive_failure": False if sql else None,
                        }
                    )
    return pd.DataFrame(rows)


def test_sql_execution_failure_excludes_question_from_every_model_and_method() -> None:
    data = _matrix()
    failed = (
        data["dataset"].eq("spider")
        & data["example_id"].eq("spider:db:1")
        & data["pipeline"].eq("text_to_sql")
        & data["model_id"].str.startswith("Qwen")
    )
    # These are real failure semantics: model generation succeeded, but SQL never executed.
    data.loc[failed, "execution_success"] = False
    analysis = calculate_common_cohort_analysis(data)

    assert analysis.question_counts == {"spider": 1, "bird": 2}
    assert analysis.available_question_counts == {"spider": 2, "bird": 2}
    assert len(analysis.records) == 18
    assert "spider:db:1" not in set(analysis.records["example_id"])
    counts = analysis.records.groupby(["dataset", "method"])["example_id"].size()
    assert counts.loc["spider"].tolist() == [2, 2, 2]
    assert counts.loc["bird"].tolist() == [4, 4, 4]


@pytest.mark.parametrize("reason", ["missing", "context", "empty"])
def test_incomplete_direct_run_excludes_the_entire_question(reason: str) -> None:
    data = _matrix()
    index = data.index[data["example_id"].eq("spider:db:1") & data["format"].eq("json")][0]
    if reason == "missing":
        data = data.drop(index)
    elif reason == "context":
        data.loc[index, "error"] = "context_exceeded"
    else:
        data.loc[index, "has_answer"] = False
    analysis = calculate_common_cohort_analysis(data)
    assert analysis.question_counts["spider"] == 1
    assert "spider:db:1" not in set(analysis.records["example_id"])


def test_success_after_retry_is_retained_with_cumulative_cost_and_wrong_answers() -> None:
    data = _matrix()
    index = data.index[data["pipeline"].eq("text_to_sql")][0]
    data.at[index, ATTEMPT_TTFT_COLUMN] = [0.2, 0.3]
    data.at[index, ATTEMPT_GENERATION_COLUMN] = [1.0, 2.0]
    data.loc[index, ["retry_count", "generation_time_s", "total_time_s"]] = [1, 3.0, 3.1]
    data.loc[index, "input_tokens"] = 60
    analysis = calculate_common_cohort_analysis(data)
    sql = analysis.summary.set_index(["dataset", "method"]).loc[("spider", "text_to_sql")]

    assert analysis.question_counts["spider"] == 2
    assert sql["average_ttft_s"] == pytest.approx((0.5 + 3 * 0.2) / 4)
    assert sql["average_total_latency_s"] == pytest.approx((3.1 + 3 * 1.1) / 4)
    assert sql["average_input_tokens"] == 30
    assert sql["average_retries"] == 0.25
    assert not analysis.records["execution.result_match"].eq(True).any()
    assert sql["average_ttft_s"] + sql["average_generation_time_s"] + sql[
        "average_sql_execution_time_s"
    ] == pytest.approx(sql["average_total_latency_s"])


def test_pairing_and_ratios_use_the_same_question_and_model() -> None:
    analysis = calculate_common_cohort_analysis(_matrix())
    assert len(analysis.paired_comparisons) == 16
    csv_pairs = analysis.paired_comparisons[analysis.paired_comparisons["method"].eq("csv")]
    assert csv_pairs["latency_ratio"].tolist() == pytest.approx([2 / 1.1] * 8)
    assert csv_pairs["input_token_ratio"].tolist() == [5] * 8
    assert csv_pairs["sql_faster"].all()


def test_default_model_exclusion_happens_before_cohort_selection() -> None:
    data = _matrix()
    excluded = data.iloc[[0]].copy()
    excluded["model_id"] = "Qwen3.5-27B-Q4.gguf"
    excluded["error"] = "context_exceeded"
    analysis = calculate_common_cohort_analysis(pd.concat([data, excluded], ignore_index=True))
    assert analysis.model_count == 2
    assert len(analysis.records) == len(data)


def test_duplicate_run_is_rejected_instead_of_double_weighted() -> None:
    data = _matrix()
    with pytest.raises(ValueError, match="Duplicate common-cohort run"):
        calculate_common_cohort_analysis(pd.concat([data, data.iloc[[0]]], ignore_index=True))


def test_dataset_with_no_shared_successful_questions_is_rejected() -> None:
    data = _matrix()
    data.loc[data["dataset"].eq("bird"), "has_answer"] = False
    with pytest.raises(ValueError, match="No successful common question cohort for bird"):
        calculate_common_cohort_analysis(data)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1.0])
def test_invalid_metrics_cannot_silently_enter_means(value: float) -> None:
    data = _matrix()
    data["input_tokens"] = data["input_tokens"].astype(float)
    data.loc[0, "input_tokens"] = value
    with pytest.raises(ValueError):
        calculate_common_cohort_analysis(data)


def test_artifacts_show_all_methods_and_distinguish_log_and_linear_scales(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    analysis = calculate_common_cohort_analysis(_matrix())
    table = render_common_cohort_table(analysis)
    assert r"\label{tab:common-cohort-operational-efficiency}" in table
    assert "Completion does not imply correctness" in table
    assert "all correction attempts" in table
    assert "Direct / CSV" in table
    assert "Text-to-SQL" in table
    latency = plot_common_cohort_latency(analysis)
    tokens = plot_common_cohort_token_usage(analysis)
    models = plot_common_cohort_latency_by_model(analysis)
    assert len(latency.axes) == 2
    assert [text.get_text() for text in latency.legends[0].get_texts()][0] == (
        "TTFT (cumulative for SQL)"
    )
    assert len(tokens.axes) == 4
    assert [axis.get_xscale() for axis in tokens.axes] == ["log", "log", "linear", "linear"]
    assert len(models.axes) == 3  # two panels plus the colorbar
    for axis in latency.axes:
        assert len(axis.containers) == 3
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            "Text-to-SQL",
            "Direct / CSV",
            "Direct / JSON",
        ]
    for generator, filename in (
        (generate_common_cohort_latency_figure, COMMON_COHORT_LATENCY_FIGURE_FILENAME),
        (generate_common_cohort_token_figure, COMMON_COHORT_TOKENS_FIGURE_FILENAME),
        (generate_common_cohort_model_figure, COMMON_COHORT_MODEL_FIGURE_FILENAME),
    ):
        path = generator(analysis, tmp_path)
        assert path.name == filename
        assert path.read_bytes().startswith(b"%PDF-")
    assert generate_common_cohort_table(analysis, tmp_path).name == COMMON_COHORT_TABLE_FILENAME
    for figure in (latency, tokens, models):
        plt.close(figure)


def test_orchestrator_registers_all_new_artifacts(tmp_path: Path) -> None:
    report = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "database_id": f"{dataset}-db",
                "format": fmt,
                "tokenizer_family": family,
                "tokenizer_label": family,
                "row_count": 10,
                "table_count": 1,
                "tokens": 100 if fmt == "csv" else 400,
            }
            for dataset in ("spider", "bird")
            for fmt in ("csv", "json")
            for family in ("qwen-3.5", "ministral-3", "gemma-4")
        ]
    )
    outputs = generate_all(
        _matrix(),
        report,
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        coverage_validated=False,
    )
    names = {path.name for path in outputs}
    assert {
        COMMON_COHORT_TABLE_FILENAME,
        COMMON_COHORT_LATENCY_FIGURE_FILENAME,
        COMMON_COHORT_TOKENS_FIGURE_FILENAME,
        COMMON_COHORT_MODEL_FIGURE_FILENAME,
    } <= names
