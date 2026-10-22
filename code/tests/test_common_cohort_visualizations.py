"""Tests for recorded costs on an explicitly selected shared context-fit cohort."""

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
    COMMON_COHORT_METHOD_LABELS,
    COMMON_COHORT_METHOD_ORDER,
    COMMON_COHORT_TABLE_FILENAME,
    COMMON_COHORT_TOKENS_FIGURE_FILENAME,
)
from visualizations.direct_accuracy_data import METRIC_SOURCE_COLUMNS
from visualizations.direct_efficiency_data import calculate_direct_efficiency
from visualizations.text_to_sql_data import (
    ATTEMPT_GENERATION_COLUMN,
    ATTEMPT_INPUT_COLUMN,
    ATTEMPT_OUTPUT_COLUMN,
    ATTEMPT_TTFT_COLUMN,
    calculate_text_to_sql_analysis,
)


def _matrix() -> pd.DataFrame:
    rows = []
    for dataset in ("spider", "bird"):
        for question in (0, 1):
            for model in ("Qwen3.5-2B-Q4.gguf", "gemma-4-E4B-Q4.gguf"):
                for method in COMMON_COHORT_METHOD_ORDER:
                    sql = method == "text_to_sql"
                    total = {"csv": 2.0, "text_to_sql": 1.1}.get(method, 8.0)
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
                            "input_tokens": {"csv": 100, "text_to_sql": 20}.get(method, 400),
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
                            ATTEMPT_TTFT_COLUMN: [0.2] if sql else None,
                            ATTEMPT_GENERATION_COLUMN: [1.0] if sql else None,
                            ATTEMPT_INPUT_COLUMN: [20] if sql else None,
                            ATTEMPT_OUTPUT_COLUMN: [10] if sql else None,
                            "predicted_sql": "SELECT 1" if sql else None,
                            "definitive_failure": False if sql else None,
                        }
                    )
    return pd.DataFrame(rows)


def test_sql_execution_failure_and_blank_outputs_retain_every_question_and_cost() -> None:
    data = _matrix()
    failed = (
        data["dataset"].eq("spider")
        & data["example_id"].eq("spider:db:1")
        & data["pipeline"].eq("text_to_sql")
        & data["model_id"].str.startswith("Qwen")
    )
    # These are real failure semantics: model generation succeeded, but SQL never executed.
    data.loc[failed, "execution_success"] = False
    data.loc[failed, list(METRIC_SOURCE_COLUMNS.values())] = 0.0
    data.loc[failed, "has_answer"] = False
    data.loc[data["pipeline"].eq("direct"), "has_answer"] = False
    analysis = calculate_common_cohort_analysis(data)

    assert analysis.question_counts == {"spider": 2, "bird": 2}
    assert len(analysis.records) == len(data)
    assert "spider:db:1" in set(analysis.records["example_id"])
    counts = analysis.records.groupby(["dataset", "method"])["example_id"].size()
    assert counts.loc["spider"].tolist() == [4] * 7
    assert counts.loc["bird"].tolist() == [4] * 7
    sql = analysis.records[analysis.records["pipeline"].eq("text_to_sql")]
    assert sql["execution_success"].eq(False).sum() == 1
    assert sql["input_tokens"].sum() == 8 * 20


@pytest.mark.parametrize("method", ["json", "text_to_sql"])
def test_missing_selected_run_raises_instead_of_shrinking_the_workload(method: str) -> None:
    data = _matrix()
    methods = data["format"].fillna("text_to_sql")
    index = data.index[data["example_id"].eq("spider:db:1") & methods.eq(method)][0]
    with pytest.raises(ValueError, match="identical question/model sets"):
        calculate_common_cohort_analysis(data.drop(index))


@pytest.mark.parametrize("success", [True, False])
def test_retries_retain_all_costs_for_successes_and_terminal_failures(success: bool) -> None:
    data = _matrix()
    index = data.index[data["pipeline"].eq("text_to_sql")][0]
    data.at[index, ATTEMPT_TTFT_COLUMN] = [0.2, 0.3]
    data.at[index, ATTEMPT_GENERATION_COLUMN] = [1.0, 2.0]
    data.at[index, ATTEMPT_INPUT_COLUMN] = [20, 40]
    data.at[index, ATTEMPT_OUTPUT_COLUMN] = [10, 0]
    data.loc[index, "execution_success"] = success
    if not success:
        data.loc[index, list(METRIC_SOURCE_COLUMNS.values())] = 0.0
    data.loc[index, ["retry_count", "generation_time_s", "total_time_s"]] = [1, 3.0, 3.1]
    data.loc[index, "input_tokens"] = 60
    analysis = calculate_common_cohort_analysis(data)
    sql = analysis.summary.set_index(["dataset", "method"]).loc[("spider", "text_to_sql")]

    assert analysis.question_counts["spider"] == 2
    assert sql["average_ttft_s"] == pytest.approx((0.5 + 3 * 0.2) / 4)
    assert sql["average_total_latency_s"] == pytest.approx((3.1 + 3 * 1.1) / 4)
    assert sql["average_input_tokens"] == 30
    assert sql["average_retries"] == 0.25
    assert sql["average_ttft_s"] + sql["average_generation_time_s"] + sql[
        "average_sql_execution_time_s"
    ] == pytest.approx(sql["average_total_latency_s"])


def test_pairing_and_ratios_use_the_same_question_and_model() -> None:
    analysis = calculate_common_cohort_analysis(_matrix())
    assert len(analysis.paired_comparisons) == 48
    csv_pairs = analysis.paired_comparisons[analysis.paired_comparisons["method"].eq("csv")]
    assert csv_pairs["latency_ratio"].tolist() == pytest.approx([2 / 1.1] * 8)
    assert csv_pairs["input_token_ratio"].tolist() == [5] * 8
    assert csv_pairs["sql_faster"].all()


def test_default_model_exclusion_happens_before_matrix_validation() -> None:
    data = _matrix()
    excluded = data.iloc[[0]].copy()
    excluded["model_id"] = "Qwen3.5-27B-Q4.gguf"
    excluded["error"] = "context_exceeded"
    analysis = calculate_common_cohort_analysis(pd.concat([data, excluded], ignore_index=True))
    assert analysis.model_count == 2
    assert len(analysis.records) == len(data)


def test_duplicate_run_is_rejected_instead_of_double_weighted() -> None:
    data = _matrix()
    with pytest.raises(ValueError, match="Duplicate operational run"):
        calculate_common_cohort_analysis(pd.concat([data, data.iloc[[0]]], ignore_index=True))


def test_inconsistent_database_identity_is_rejected() -> None:
    data = _matrix()
    data.loc[0, "db_id"] = "other-db"
    with pytest.raises(ValueError, match="inconsistent database identities"):
        calculate_common_cohort_analysis(data)


def test_direct_and_common_calculations_use_identical_records_and_numbers() -> None:
    data = _matrix()
    data.loc[data["pipeline"].eq("direct"), "has_answer"] = False
    data.loc[data["pipeline"].eq("text_to_sql"), "execution_success"] = False
    data.loc[data["pipeline"].eq("text_to_sql"), list(METRIC_SOURCE_COLUMNS.values())] = 0.0
    direct = calculate_direct_efficiency(data)
    common = calculate_common_cohort_analysis(data)
    sql = calculate_text_to_sql_analysis(data)
    assert direct.question_counts == common.question_counts == sql.question_counts
    keys = ["db_id", "example_id", "model_id"]
    for _dataset, records in common.records.groupby("dataset"):
        sets = [set(rows[keys].itertuples(index=False, name=None))
                for _, rows in records.groupby("method")]
        assert all(pairs == sets[0] for pairs in sets)
    metrics = ["average_input_tokens", "average_output_tokens", "average_ttft_s",
               "average_generation_time_s", "average_total_latency_s"]
    left = direct.summary.set_index(["dataset", "format"])[metrics].sort_index()
    right = common.summary[common.summary["method"].ne("text_to_sql")].rename(
        columns={"method": "format"}
    ).set_index(["dataset", "format"])[metrics].sort_index()
    pd.testing.assert_frame_equal(left, right)
    sql_means = common.summary[common.summary["method"].eq("text_to_sql")].set_index("dataset")
    pd.testing.assert_series_equal(
        sql_means["average_total_latency_s"].sort_index(),
        sql.dataset_summary.set_index("dataset")["average_total_latency_s"].sort_index(),
    )


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
    assert "recorded context-fit" in table
    assert "SQL failures" in table
    assert "all correction attempts" in table
    assert "Direct / CSV" in table
    assert "Text-to-SQL" in table
    latency = plot_common_cohort_latency(analysis)
    tokens = plot_common_cohort_token_usage(analysis)
    assert len(latency.axes) == 2
    assert [text.get_text() for text in latency.legends[0].get_texts()][0] == (
        "TTFT (cumulative for SQL)"
    )
    assert len(tokens.axes) == 4
    assert [axis.get_xscale() for axis in tokens.axes] == ["log", "log", "linear", "linear"]
    for axis in latency.axes:
        assert len(axis.containers) == 3
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            COMMON_COHORT_METHOD_LABELS[method] for method in COMMON_COHORT_METHOD_ORDER
        ]
    for generator, filename in (
        (generate_common_cohort_latency_figure, COMMON_COHORT_LATENCY_FIGURE_FILENAME),
        (generate_common_cohort_token_figure, COMMON_COHORT_TOKENS_FIGURE_FILENAME),
    ):
        path = generator(analysis, tmp_path)
        assert path.name == filename
        assert path.read_bytes().startswith(b"%PDF-")
    assert generate_common_cohort_table(analysis, tmp_path).name == COMMON_COHORT_TABLE_FILENAME
    for figure in (latency, tokens):
        plt.close(figure)
