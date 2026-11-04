"""Tests for Text-to-SQL retrieval and operational artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import numpy as np
import pandas as pd

from visualizations.config import (
    TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME,
    TEXT_TO_SQL_LATENCY_FIGURE_FILENAME,
    TEXT_TO_SQL_OPERATIONAL_TABLE_FILENAME,
    TEXT_TO_SQL_RETRIEVAL_TABLE_FILENAME,
    TEXT_TO_SQL_RETRY_FIGURE_FILENAME,
    TEXT_TO_SQL_RETRY_TABLE_FILENAME,
    TEXT_TO_SQL_TOKEN_FIGURE_FILENAME,
)
from visualizations.text_to_sql_accuracy_by_model_size import (
    generate_text_to_sql_accuracy_figure,
    plot_text_to_sql_accuracy,
)
from visualizations.text_to_sql_data import calculate_text_to_sql_analysis
from visualizations.text_to_sql_error_correction import (
    generate_error_correction_figure,
    plot_error_correction_performance,
)
from visualizations.text_to_sql_error_correction_table import (
    generate_error_correction_table,
    render_error_correction_table,
)
from visualizations.text_to_sql_latency import (
    generate_text_to_sql_latency_figure,
    plot_text_to_sql_latency,
)
from visualizations.text_to_sql_operational_efficiency_table import (
    generate_text_to_sql_operational_table,
    render_text_to_sql_operational_table,
)
from visualizations.text_to_sql_retrieval_accuracy_table import (
    generate_text_to_sql_retrieval_table,
    render_text_to_sql_retrieval_table,
)
from visualizations.text_to_sql_retry_data import calculate_text_to_sql_retry_analysis
from visualizations.text_to_sql_token_usage import (
    generate_text_to_sql_token_figure,
    plot_text_to_sql_token_usage,
)


def _record(
    *,
    dataset: str = "spider",
    model: str = "Qwen3.5-2B-Q4.gguf",
    example_id: str = "spider:db:0",
    exact: float = 0.5,
    bert: float = 0.6,
    rouge: float = 0.7,
    input_tokens: int = 100,
    output_tokens: int = 20,
    attempt_ttft: list[float] | None = None,
    attempt_generation: list[float] | None = None,
    model_time: float = 2.0,
    sql_time: float = 0.1,
    retry_count: int = 0,
    execution_success: bool = True,
) -> dict[str, object]:
    attempt_ttft = attempt_ttft or [0.5]
    attempt_generation = attempt_generation or [model_time]
    return {
        "dataset": dataset,
        "db_id": "db",
        "example_id": example_id,
        "model_id": model,
        "pipeline": "text_to_sql",
        "format": None,
        "error": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "time_to_first_token_s": attempt_ttft[0],
        "generation_time_s": model_time,
        "query_execution_time_s": sql_time,
        "total_time_s": model_time + sql_time,
        "retry_count": retry_count,
        "execution_success": execution_success,
        "scores.normalized_exact_match": exact,
        "scores.bertscore_f": bert,
        "scores.rouge_l_f": rouge,
        "attempts.llm_metrics.time_to_first_token_s": attempt_ttft,
        "attempts.llm_metrics.total_generation_time_s": attempt_generation,
        "attempts.llm_metrics.input_tokens": [input_tokens - retry_count] + [1] * retry_count,
        "attempts.llm_metrics.output_tokens": [output_tokens - retry_count] + [1] * retry_count,
    }


def _results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("spider", "bird"):
        rows.extend(
            [
                _record(
                    dataset=dataset,
                    model="Qwen3.5-2B-Q4.gguf",
                    example_id=f"{dataset}:db:0",
                    exact=0.0,
                    bert=0.0,
                    rouge=0.0,
                    input_tokens=300,
                    output_tokens=60,
                    attempt_ttft=[0.5, 0.7],
                    attempt_generation=[1.0, 2.0],
                    model_time=3.0,
                    retry_count=1,
                    execution_success=False,
                ),
                _record(
                    dataset=dataset,
                    model="Qwen3.5-4B-Q4.gguf",
                    example_id=f"{dataset}:db:0",
                    exact=0.8,
                    bert=0.9,
                    rouge=0.7,
                    input_tokens=100,
                    output_tokens=20,
                    attempt_ttft=[0.4],
                    model_time=1.5,
                ),
                _record(
                    dataset=dataset,
                    model="gemma-4-E4B-Q4.gguf",
                    example_id=f"{dataset}:db:0",
                    exact=0.9,
                    bert=0.95,
                    rouge=0.8,
                    input_tokens=90,
                    output_tokens=15,
                    attempt_ttft=[0.3],
                    model_time=1.0,
                ),
            ]
        )
    return pd.DataFrame(rows)


def test_analysis_includes_failed_accuracy_zeroes_and_accumulates_retry_ttft() -> None:
    analysis = calculate_text_to_sql_analysis(_results())
    spider = analysis.dataset_summary.set_index("dataset").loc["spider"]
    qwen_2b = analysis.model_summary[
        analysis.model_summary["model_id"].eq("Qwen3.5-2B-Q4.gguf")
        & analysis.model_summary["dataset"].eq("spider")
    ].iloc[0]

    assert spider["normalized_exact_match"] == pytest.approx((0.0 + 0.8 + 0.9) / 3)
    assert spider["execution_success_rate"] == pytest.approx(2 / 3)
    assert "result_match_rate" not in analysis.dataset_summary.columns
    assert "result_match_rate" not in analysis.model_summary.columns
    assert qwen_2b["average_cumulative_ttft_s"] == pytest.approx(1.2)
    assert qwen_2b["average_generation_time_s"] == pytest.approx(1.8)
    assert qwen_2b["average_total_latency_s"] == pytest.approx(3.1)
    assert qwen_2b["average_retries"] == pytest.approx(1.0)


def test_analysis_rejects_inconsistent_total_latency() -> None:
    row = _record()
    row["total_time_s"] = 9.0

    with pytest.raises(ValueError, match="inconsistent total latency"):
        calculate_text_to_sql_analysis(pd.DataFrame([row]))


def test_text_to_sql_tables_report_accuracy_execution_and_operations(tmp_path: Path) -> None:
    analysis = calculate_text_to_sql_analysis(_results())
    retrieval = render_text_to_sql_retrieval_table(analysis)
    operational = render_text_to_sql_operational_table(analysis)

    assert "NEM" in retrieval
    assert "BERTScore & ROUGE-L" in retrieval
    for table in (retrieval, operational):
        assert r"\parbox" not in table
        assert "Note:" not in table
    assert "F1" not in retrieval
    assert "answerable cohort" in retrieval
    assert "answerable cohort" in operational
    assert "Executed" in retrieval
    assert "Correct result" not in retrieval
    assert "Input tokens" in operational
    assert "SQL (s)" in operational
    assert "Retries" in operational

    tables_dir = tmp_path / "tables"
    retrieval_path = generate_text_to_sql_retrieval_table(analysis, tables_dir)
    operational_path = generate_text_to_sql_operational_table(analysis, tables_dir)
    assert retrieval_path == tables_dir / TEXT_TO_SQL_RETRIEVAL_TABLE_FILENAME
    assert operational_path == tables_dir / TEXT_TO_SQL_OPERATIONAL_TABLE_FILENAME


def test_retry_analysis_reports_terminal_stage_execution_and_added_time() -> None:
    sql_analysis = calculate_text_to_sql_analysis(_results())
    retry_analysis = calculate_text_to_sql_retry_analysis(sql_analysis)
    spider = retry_analysis.outcome_summary[
        retry_analysis.outcome_summary["dataset"].eq("spider")
    ].set_index("terminal_outcome")
    stages = retry_analysis.retry_stage_summary[
        retry_analysis.retry_stage_summary["dataset"].eq("spider")
    ].set_index("retry_index")

    assert spider.loc["attempt_1", "run_count"] == 2
    assert spider.loc["failure", "run_count"] == 1
    assert spider.loc["failure", "total_retry_generation_s"] == pytest.approx(2.0)
    assert "correct_result_count" not in retry_analysis.outcome_summary.columns
    assert "correct_gain_pp" not in retry_analysis.gain_summary.columns
    assert stages.loc[1, "run_count"] == 1
    assert stages.loc[1, "total_generation_s"] == pytest.approx(2.0)
    assert retry_analysis.total_retry_generation_s == pytest.approx(4.0)

    successful_retry = _results()
    successful_retry.loc[0, "execution_success"] = True
    retry_gain = calculate_text_to_sql_retry_analysis(
        calculate_text_to_sql_analysis(successful_retry)
    ).gain_summary.set_index("dataset")
    assert retry_gain.loc["spider", "execution_gain_pp"] == pytest.approx(100 / 3)


def test_error_correction_table(tmp_path: Path) -> None:
    retry_analysis = calculate_text_to_sql_retry_analysis(
        calculate_text_to_sql_analysis(_results())
    )
    latex = render_error_correction_table(retry_analysis)

    assert "First attempt" in latex
    assert "Complete failure" in latex
    assert "/ eligible run" not in latex
    assert "Mean added generation (s) / run entering correction" in latex
    assert "Recovery / initially failed runs" in latex
    assert "Mean retries / run entering correction" in latex
    assert r"\parbox" not in latex
    assert "Note:" not in latex
    assert "Correct results" not in latex
    table_path = generate_error_correction_table(retry_analysis, tmp_path / "tables")
    assert table_path.name == TEXT_TO_SQL_RETRY_TABLE_FILENAME


def test_error_correction_figure(tmp_path: Path) -> None:
    retry_analysis = calculate_text_to_sql_retry_analysis(
        calculate_text_to_sql_analysis(_results())
    )
    figure, axes = plot_error_correction_performance(retry_analysis)
    assert [axis.get_title() for axis in axes] == [
        "Added generation time by terminal outcome",
        "Added generation time by retry stage",
    ]
    # Both decompositions sum to the same added seconds per eligible run.
    for axis in axes:
        assert axis.get_ylabel() == "Additional seconds per answerable run"
        assert sum(patch.get_height() for patch in axis.patches) == pytest.approx(4 / 3)

    figure_path = generate_error_correction_figure(retry_analysis, tmp_path / "figures")
    assert figure_path.name == TEXT_TO_SQL_RETRY_FIGURE_FILENAME
    assert figure_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_text_to_sql_figures_show_metrics_tokens_latency_and_retries(tmp_path: Path) -> None:
    analysis = calculate_text_to_sql_analysis(_results())
    accuracy_figure, accuracy_axes = plot_text_to_sql_accuracy(analysis)
    token_figure, token_axes = plot_text_to_sql_token_usage(analysis)
    latency_figure, latency_axes = plot_text_to_sql_latency(analysis)

    assert len(accuracy_axes) == 3
    assert all(len(row) == 2 for row in accuracy_axes)
    assert [axis.get_title() for axis in accuracy_axes[0]] == ["Spider", "BIRD"]
    assert [row[0].get_ylabel() for row in accuracy_axes] == [
        "NEM\nMean score",
        "BERTScore\nMean score",
        "ROUGE-L\nMean score",
    ]
    assert [axis.get_title() for axis in token_axes[0]] == [
        "Average input tokens",
        "Average output tokens",
    ]
    assert all(len(axis.containers) == 3 for axis in latency_axes)
    for axis in latency_axes:
        formatter = axis.yaxis.get_major_formatter()
        assert formatter(0.5, 0) == r"\num{0.5}"
        assert formatter(1.5, 1) == r"\num{1.5}"
    assert [text.get_text() for text in latency_figure.legends[0].get_texts()][0] == (
        "Cumulative TTFT"
    )
    assert any(r"\num{1.00} retries" in text.get_text() for text in latency_axes[0].texts)

    figures_dir = tmp_path / "figures"
    accuracy_path = generate_text_to_sql_accuracy_figure(analysis, figures_dir)
    token_path = generate_text_to_sql_token_figure(analysis, figures_dir)
    latency_path = generate_text_to_sql_latency_figure(analysis, figures_dir)
    assert accuracy_path == figures_dir / TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME
    assert token_path == figures_dir / TEXT_TO_SQL_TOKEN_FIGURE_FILENAME
    assert latency_path == figures_dir / TEXT_TO_SQL_LATENCY_FIGURE_FILENAME
    assert all(path.stat().st_size > 0 for path in (accuracy_path, token_path, latency_path))

    import matplotlib.pyplot as plt

    plt.close(accuracy_figure)
    plt.close(token_figure)
    plt.close(latency_figure)


@pytest.mark.parametrize("column,value,message", [
    ("input_tokens", float("inf"), "non-finite"),
    ("query_execution_time_s", None, "missing"),
    ("output_tokens", -1, "negative"),
    ("retry_count", 0.5, "integer"),
    ("retry_count", 4, "three-retry"),
    ("execution_success", None, "boolean"),
    ("execution_success", "False", "boolean"),
    ("attempts.llm_metrics.total_generation_time_s", [float("inf")], "generation data"),
    ("attempts.llm_metrics.time_to_first_token_s", [0.1, 0.2], "TTFT counts"),
    ("attempts.llm_metrics.time_to_first_token_s", [3.0], "TTFT exceeds"),
    ("time_to_first_token_s", 0.1, "initial TTFT"),
    ("db_id", "", "identities"),
    ("db_id", None, "identities"),
    ("attempts.llm_metrics.input_tokens", [99], "accumulated input_tokens"),
    ("attempts.llm_metrics.output_tokens", [None], "valid attempt output_tokens"),
])
def test_invalid_operational_measurements_stop_analysis(column, value, message) -> None:
    row = _record()
    row[column] = value
    with pytest.raises(ValueError, match=message):
        calculate_text_to_sql_analysis(pd.DataFrame([row]))


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("column", [
    "input_tokens", "output_tokens", "time_to_first_token_s", "generation_time_s",
    "query_execution_time_s", "total_time_s", "retry_count",
    "attempts.llm_metrics.time_to_first_token_s",
    "attempts.llm_metrics.total_generation_time_s",
    "attempts.llm_metrics.input_tokens", "attempts.llm_metrics.output_tokens",
])
def test_boolean_operational_measurements_are_rejected(column, value) -> None:
    row = _record()
    row[column] = [value] if column.startswith("attempts.") else value
    with pytest.raises(ValueError, match="boolean|lack"):
        calculate_text_to_sql_analysis(pd.DataFrame([row]))


def test_failed_runs_keep_zero_scores_without_dropping_work_or_selecting_questions() -> None:
    rows = [
        _record(example_id="spider:db:empty-reference", execution_success=False,
                exact=0, bert=0, rouge=0),
        _record(example_id="spider:db:nonempty-reference"),
    ]
    rows[0]["error"] = "context_exceeded"
    rows[0]["raw_output"] = ""
    analysis = calculate_text_to_sql_analysis(pd.DataFrame(rows))
    assert analysis.run_count == 2
    assert analysis.question_counts == {"spider": 2}
    failed = analysis.records.iloc[0]
    assert failed["scores.normalized_exact_match"] == 0
    assert failed["scores.bertscore_f"] == 0
    assert failed["scores.rouge_l_f"] == 0
    assert analysis.dataset_summary.iloc[0]["average_input_tokens"] == 100
    assert analysis.dataset_summary.iloc[0]["average_total_latency_s"] == pytest.approx(2.1)


def test_mixed_pipeline_boolean_column_is_validated_after_sql_selection() -> None:
    sql = _record(execution_success=False, exact=0, bert=0, rouge=0)
    direct = {**_record(), "pipeline": "direct", "execution_success": None}
    analysis = calculate_text_to_sql_analysis(pd.DataFrame([sql, direct]))
    assert analysis.run_count == 1
    assert analysis.dataset_summary.iloc[0]["normalized_exact_match"] == 0


def test_retry_denominators_include_early_failures_and_all_retry_costs() -> None:
    # One success at each stage, one exhausted failure, and one fatal initial failure.
    rows = []
    for index, (attempts, success) in enumerate(
        [(1, True), (2, True), (3, True), (4, True), (4, False), (1, False)]
    ):
        row = _record(
            example_id=f"spider:db:{index}", execution_success=success,
            exact=0, bert=0, rouge=0,
            retry_count=attempts - 1, attempt_ttft=[0.1] * attempts,
            attempt_generation=[2.0] * attempts, model_time=2.0 * attempts,
        )
        row["db_id"] = f"db{index % 2}"
        rows.append(row)
    analysis = calculate_text_to_sql_retry_analysis(
        calculate_text_to_sql_analysis(pd.DataFrame(rows))
    )
    summary = analysis.gain_summary.iloc[0]
    assert summary["run_count"] == 6
    assert summary["initially_failed_run_count"] == 5
    assert summary["retried_run_count"] == 4
    assert summary["first_attempt_success_rate"] == pytest.approx(1 / 6)
    assert summary["final_success_rate"] == pytest.approx(4 / 6)
    assert summary["recovery_rate"] == pytest.approx(3 / 5)
    assert summary["execution_gain_pp"] == pytest.approx(50)
    assert summary["average_retries"] == pytest.approx(9 / 6)
    assert summary["average_retries_per_retried_run"] == pytest.approx(9 / 4)
    assert summary["added_seconds"] == pytest.approx(18 / 6)
    assert summary["added_seconds_per_retried_run"] == pytest.approx(18 / 4)
    assert summary["execution_gain_pp_low"] == pytest.approx(100 / 3)
    assert summary["execution_gain_pp_high"] == pytest.approx(200 / 3)
    outcomes = analysis.outcome_summary.query("dataset == 'spider'").set_index("terminal_outcome")
    assert list(outcomes["run_count"]) == [1, 1, 1, 1, 2]
    assert outcomes.loc["failure", "total_retry_generation_s"] == 6
    assert outcomes.loc["failure", "total_retries"] == 3
    assert outcomes.loc["failure", "average_retries"] == 1.5


def test_no_initial_failures_has_undefined_conditional_rates_not_zero() -> None:
    analysis = calculate_text_to_sql_retry_analysis(
        calculate_text_to_sql_analysis(pd.DataFrame([_record()]))
    )
    summary = analysis.gain_summary.iloc[0]
    assert summary["first_attempt_success_rate"] == 1
    assert summary["added_seconds"] == 0
    assert pd.isna(summary["recovery_rate"])
    assert pd.isna(summary["added_seconds_per_retried_run"])
    assert "---" in render_error_correction_table(analysis)
    assert set(analysis.outcome_summary["dataset"]) == {"spider"}
    assert set(analysis.retry_stage_summary["dataset"]) == {"spider"}
    assert analysis.retry_stage_summary["average_generation_s_when_used"].isna().all()
    assert analysis.retry_stage_summary["average_generation_s_per_retried_run"].isna().all()
    absent = analysis.outcome_summary.query("terminal_outcome != 'attempt_1'")
    assert absent["average_retries"].isna().all()
    assert absent["average_retry_generation_s"].isna().all()


@pytest.mark.parametrize("score", [0.5, None, float("inf")])
def test_failure_scores_are_validated_without_rewriting_stored_scores(score) -> None:
    row = _record(execution_success=False, exact=0, bert=0, rouge=0)
    row["scores.bertscore_f"] = score
    frame = pd.DataFrame([row])
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="accuracy scores"):
        calculate_text_to_sql_analysis(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_conditional_bootstrap_keeps_databases_with_no_correction() -> None:
    rows = []
    for index, retries in enumerate([0, 1, 2]):
        row = _record(
            example_id=f"spider:db{index}:0", retry_count=retries,
            attempt_ttft=[0.1] * (retries + 1),
            attempt_generation=[2.0] * (retries + 1), model_time=2 * (retries + 1),
        )
        row["db_id"] = f"db{index}"
        rows.append(row)
    analysis = calculate_text_to_sql_retry_analysis(
        calculate_text_to_sql_analysis(pd.DataFrame(rows))
    )
    summary = analysis.gain_summary.iloc[0]
    assert summary["recovery_rate"] == 1
    assert summary["average_retries_per_retried_run"] == 1.5
    assert summary["added_seconds_per_retried_run"] == 3
    # Resampling all three eligible clusters includes all-db0 draws with a zero
    # denominator. Dropping db0 would incorrectly produce finite conditional CIs.
    for metric in ("recovery_rate", "average_retries_per_retried_run",
                   "added_seconds_per_retried_run"):
        assert pd.isna(summary[f"{metric}_low"])
        assert pd.isna(summary[f"{metric}_high"])
    stage = analysis.retry_stage_summary.set_index("retry_index").loc[1]
    assert stage["average_generation_s_per_all_runs"] == pytest.approx(4 / 3)
    assert stage["average_generation_s_per_all_runs_low"] == 0
    assert stage["average_generation_s_per_all_runs_high"] == 2
    assert stage["average_generation_s_per_retried_run"] == 2
    assert pd.isna(stage["average_generation_s_per_retried_run_low"])
    assert pd.isna(stage["average_generation_s_per_retried_run_high"])


def test_paired_model_effects_and_database_uncertainty_are_preserved() -> None:
    rows = []
    for index, baseline in enumerate([0.1, 0.5]):
        for model, gain in [("Qwen3.5-2B-Q4.gguf", 0), ("Qwen3.5-4B-Q4.gguf", 0.2)]:
            row = _record(example_id=f"spider:db{index}:0", model=model, exact=baseline + gain)
            row["db_id"] = f"db{index}"
            rows.append(row)
    summary = calculate_text_to_sql_analysis(pd.DataFrame(rows)).model_summary
    larger = summary.set_index("model_id").loc["Qwen3.5-4B-Q4.gguf"]
    assert larger["normalized_exact_match_effect"] == pytest.approx(0.2)
    assert larger["normalized_exact_match_effect_low"] == pytest.approx(0.2)
    assert larger["normalized_exact_match_effect_high"] == pytest.approx(0.2)
    assert larger["normalized_exact_match_low"] == pytest.approx(0.3)
    assert larger["normalized_exact_match_high"] == pytest.approx(0.7)
