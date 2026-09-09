"""Tests for Text-to-SQL retrieval and operational artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

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
    result_match: bool = True,
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
        "execution.result_match": result_match,
        "attempts.llm_metrics.time_to_first_token_s": attempt_ttft,
        "attempts.llm_metrics.total_generation_time_s": attempt_generation,
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
                    result_match=False,
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
    assert "Executed" in retrieval
    assert "Correct result" in retrieval
    assert "Input tokens" in operational
    assert "SQL (s)" in operational
    assert "Retries" in operational

    tables_dir = tmp_path / "tables"
    retrieval_path = generate_text_to_sql_retrieval_table(analysis, tables_dir)
    operational_path = generate_text_to_sql_operational_table(analysis, tables_dir)
    assert retrieval_path == tables_dir / TEXT_TO_SQL_RETRIEVAL_TABLE_FILENAME
    assert operational_path == tables_dir / TEXT_TO_SQL_OPERATIONAL_TABLE_FILENAME


def test_retry_analysis_reports_terminal_stage_correctness_and_added_time() -> None:
    sql_analysis = calculate_text_to_sql_analysis(_results())
    retry_analysis = calculate_text_to_sql_retry_analysis(sql_analysis)
    spider = retry_analysis.outcome_summary[
        retry_analysis.outcome_summary["dataset"].eq("spider")
    ].set_index("terminal_outcome")
    stages = retry_analysis.retry_stage_summary[
        retry_analysis.retry_stage_summary["dataset"].eq("spider")
    ].set_index("retry_index")

    assert spider.loc["attempt_1", "run_count"] == 2
    assert spider.loc["attempt_1", "correct_result_count"] == 2
    assert spider.loc["failure", "run_count"] == 1
    assert spider.loc["failure", "total_retry_generation_s"] == pytest.approx(2.0)
    assert stages.loc[1, "run_count"] == 1
    assert stages.loc[1, "total_generation_s"] == pytest.approx(2.0)
    assert retry_analysis.total_retry_generation_s == pytest.approx(4.0)


def test_error_correction_table_and_figure(tmp_path: Path) -> None:
    retry_analysis = calculate_text_to_sql_retry_analysis(
        calculate_text_to_sql_analysis(_results())
    )
    latex = render_error_correction_table(retry_analysis)
    figure, axes = plot_error_correction_performance(retry_analysis)

    assert "First attempt" in latex
    assert "Complete failure" in latex
    assert "Added time (s)" in latex
    assert [axis.get_title() for axis in axes] == [
        "Terminal query-execution outcome",
        "Additional model time by retry stage",
    ]

    table_path = generate_error_correction_table(retry_analysis, tmp_path / "tables")
    figure_path = generate_error_correction_figure(retry_analysis, tmp_path / "figures")
    assert table_path.name == TEXT_TO_SQL_RETRY_TABLE_FILENAME
    assert figure_path.name == TEXT_TO_SQL_RETRY_FIGURE_FILENAME
    assert figure_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_text_to_sql_figures_show_metrics_tokens_latency_and_retries(tmp_path: Path) -> None:
    analysis = calculate_text_to_sql_analysis(_results())
    accuracy_figure, accuracy_axes = plot_text_to_sql_accuracy(analysis)
    token_figure, token_axes = plot_text_to_sql_token_usage(analysis)
    latency_figure, latency_axes = plot_text_to_sql_latency(analysis)

    assert len(accuracy_axes) == 2
    assert all(len(row) == 3 for row in accuracy_axes)
    assert [axis.get_title() for axis in accuracy_axes[0]] == [
        "Normalized Exact Match",
        "BERTScore F1",
        "ROUGE-L F1",
    ]
    assert [axis.get_title() for axis in token_axes[0]] == [
        "Average input tokens",
        "Average output tokens",
    ]
    assert all(len(axis.containers) == 3 for axis in latency_axes)
    assert any("1.00 retries" in text.get_text() for text in latency_axes[0].texts)

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
