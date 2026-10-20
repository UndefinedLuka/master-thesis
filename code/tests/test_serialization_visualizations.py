"""Tests for Chapter 5 serialization-efficiency artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd

from visualizations.config import (
    DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME,
    DIRECT_ACCURACY_METRIC_ORDER,
    DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME,
    DIRECT_DATABASE_SIZE_FIGURE_FILENAMES,
    DIRECT_LATENCY_FIGURE_FILENAME,
    DIRECT_OPERATIONAL_TABLE_FILENAME,
    DIRECT_TOKEN_USAGE_FIGURE_FILENAME,
    EXPERIMENT_OUTCOMES_TABLE_FILENAME,
    SERIALIZATION_EFFICIENCY_TABLE_FILENAME,
    TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME,
    TEXT_TO_SQL_LATENCY_FIGURE_FILENAME,
    TEXT_TO_SQL_OPERATIONAL_TABLE_FILENAME,
    TEXT_TO_SQL_RETRIEVAL_TABLE_FILENAME,
    TEXT_TO_SQL_RETRY_FIGURE_FILENAME,
    TEXT_TO_SQL_RETRY_TABLE_FILENAME,
    TEXT_TO_SQL_TOKEN_FIGURE_FILENAME,
    TOKENIZER_COMPARISON_FIGURE_FILENAME,
)
from visualizations.generate_all import main
from visualizations.serialization_data import load_token_report, validate_complete_coverage
from visualizations.serialization_efficiency_table import (
    calculate_serialization_efficiency,
    generate_serialization_efficiency_table,
    render_serialization_efficiency_table,
)
from visualizations.tokenizer_family_comparison import (
    calculate_tokenizer_overhead,
    generate_tokenizer_comparison_figure,
    plot_tokenizer_comparison,
)

FAMILIES = (
    ("qwen-3.5", "Qwen 3.5"),
    ("ministral-3", "Ministral 3"),
    ("gemma-4", "Gemma 4"),
)


def _record(
    *,
    dataset: str,
    database_id: str,
    serialization_format: str,
    family: str,
    label: str,
    tokens: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "database_id": database_id,
        "format": serialization_format,
        "tokenizer_family": family,
        "tokenizer_label": label,
        "tokenizer_model": f"{family}.gguf",
        "table_count": 2,
        "row_count": 100,
        "tokens": tokens,
    }


def _complete_small_report() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, database_ids in (("spider", ("s1", "s2")), ("bird", ("b1",))):
        for database_index, database_id in enumerate(database_ids, start=1):
            for format_index, serialization_format in enumerate(("csv", "json"), start=1):
                for family_index, (family, label) in enumerate(FAMILIES, start=1):
                    rows.append(
                        _record(
                            dataset=dataset,
                            database_id=database_id,
                            serialization_format=serialization_format,
                            family=family,
                            label=label,
                            tokens=100 * database_index * format_index + family_index,
                        )
                    )
    return pd.DataFrame(rows)


def test_validate_complete_coverage_accepts_full_cartesian_product() -> None:
    report = _complete_small_report()

    coverage = validate_complete_coverage(
        report,
        expected_database_counts={"spider": 2, "bird": 1},
        expected_formats=("csv", "json"),
        expected_families=tuple(family for family, _ in FAMILIES),
    )

    assert coverage.database_counts == {"spider": 2, "bird": 1}
    assert coverage.format_count == 2
    assert coverage.tokenizer_family_count == 3
    assert coverage.record_count == 18


def test_validate_complete_coverage_rejects_missing_family_measurement() -> None:
    report = _complete_small_report().iloc[:-1].copy()

    with pytest.raises(ValueError, match="Incomplete serialization-token coverage"):
        validate_complete_coverage(
            report,
            expected_database_counts={"spider": 2, "bird": 1},
            expected_formats=("csv", "json"),
            expected_families=tuple(family for family, _ in FAMILIES),
        )


def test_average_tokens_and_relative_baseline_are_dataset_specific() -> None:
    report = pd.DataFrame(
        [
            _record(
                dataset="spider",
                database_id="s1",
                serialization_format="csv",
                family="qwen-3.5",
                label="Qwen 3.5",
                tokens=100,
            ),
            _record(
                dataset="spider",
                database_id="s1",
                serialization_format="json",
                family="qwen-3.5",
                label="Qwen 3.5",
                tokens=250,
            ),
            _record(
                dataset="bird",
                database_id="b1",
                serialization_format="csv",
                family="qwen-3.5",
                label="Qwen 3.5",
                tokens=300,
            ),
            _record(
                dataset="bird",
                database_id="b1",
                serialization_format="json",
                family="qwen-3.5",
                label="Qwen 3.5",
                tokens=450,
            ),
        ]
    )

    summary = calculate_serialization_efficiency(report).average_tokens.set_index(
        ["dataset", "format"]
    )

    assert summary.loc[("spider", "csv"), "mean_tokens"] == 100
    assert summary.loc[("spider", "json"), "relative_to_lowest"] == pytest.approx(2.5)
    assert summary.loc[("bird", "csv"), "relative_to_lowest"] == pytest.approx(1.0)
    assert summary.loc[("bird", "json"), "relative_to_lowest"] == pytest.approx(1.5)


def test_database_fit_includes_payload_equal_to_context_limit() -> None:
    report = pd.DataFrame(
        [
            _record(
                dataset="spider",
                database_id=database_id,
                serialization_format="csv",
                family="qwen-3.5",
                label="Qwen 3.5",
                tokens=tokens,
            )
            for database_id, tokens in (("fits", 128), ("too-large", 129))
        ]
    )

    summary = calculate_serialization_efficiency(
        report,
        context_window_tokens=128,
    ).database_fit.iloc[0]

    assert summary["fitting_databases"] == 1
    assert summary["total_databases"] == 2
    assert summary["fit_percentage"] == pytest.approx(50.0)


def test_tokenizer_overhead_is_paired_to_best_family_per_payload() -> None:
    report = pd.DataFrame(
        [
            _record(
                dataset="spider",
                database_id=database_id,
                serialization_format="csv",
                family=family,
                label=label,
                tokens=tokens,
            )
            for database_id, family, label, tokens in (
                ("small", "qwen-3.5", "Qwen 3.5", 100),
                ("small", "ministral-3", "Ministral 3", 110),
                ("small", "gemma-4", "Gemma 4", 120),
                ("large", "qwen-3.5", "Qwen 3.5", 1_000),
                ("large", "ministral-3", "Ministral 3", 1_200),
                ("large", "gemma-4", "Gemma 4", 1_100),
            )
        ]
    )

    summary = calculate_tokenizer_overhead(report).set_index("tokenizer_family")

    assert summary.loc["qwen-3.5", "median_overhead_pct"] == pytest.approx(0.0)
    assert summary.loc["ministral-3", "median_overhead_pct"] == pytest.approx(15.0)
    assert summary.loc["gemma-4", "median_overhead_pct"] == pytest.approx(15.0)
    assert summary.loc["ministral-3", "q1_overhead_pct"] == pytest.approx(12.5)
    assert summary.loc["ministral-3", "q3_overhead_pct"] == pytest.approx(17.5)


def test_rendered_table_contains_relative_and_fit_values() -> None:
    report = _complete_small_report()
    analysis = calculate_serialization_efficiency(
        report,
        context_window_tokens=250,
    )
    latex = render_serialization_efficiency_table(
        analysis,
        context_window_tokens=250,
        coverage_validated=False,
    )

    assert r"Ratio to CSV" in latex
    assert r"1.00$\times$" in latex
    assert r"\label{tab:serialization-efficiency}" in latex
    assert "at or below 250 tokens" in latex
    assert r"1/2 (50.0\%)" in latex
    assert "Coverage validation was skipped." in latex


def test_artifact_generators_each_write_one_output(tmp_path: Path) -> None:
    report = _complete_small_report()
    figure = plot_tokenizer_comparison(calculate_tokenizer_overhead(report))
    assert len(figure.axes) == 2

    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    table_path = generate_serialization_efficiency_table(
        report,
        tables_dir,
        context_window_tokens=250,
        coverage_validated=False,
    )
    figure_path = generate_tokenizer_comparison_figure(report, figures_dir)

    assert table_path == tables_dir / SERIALIZATION_EFFICIENCY_TABLE_FILENAME
    assert figure_path == figures_dir / TOKENIZER_COMPARISON_FIGURE_FILENAME
    assert figure_path.stat().st_size > 0


def test_single_orchestrator_cli_writes_every_artifact(tmp_path: Path) -> None:
    report_path = tmp_path / "serialization_tokens_by_database.csv"
    _complete_small_report().to_csv(report_path, index=False)
    results_path = tmp_path / "sample.scored.ndjson"
    direct_result = {
        "dataset": "spider",
        "pipeline": "direct",
        "format": "csv",
        "model_id": "Qwen3.5-2B-Q4_K_M.gguf",
        "example_id": "spider:db:0",
        "db_id": "s1",
        "predicted_sql": None,
        "retry_count": 0,
        "execution_success": None,
        "definitive_failure": None,
        "error": None,
        "raw_output": "answer",
        "input_tokens": 100,
        "output_tokens": 10,
        "time_to_first_token_s": 1.0,
        "generation_time_s": 3.0,
        "total_time_s": 3.0,
        "scores": {
            "normalized_exact_match": 0.5,
            "bertscore_f": 0.4,
            "rouge_l_f": 0.6,
        },
    }
    sql_result = {
        "dataset": "spider",
        "pipeline": "text_to_sql",
        "format": None,
        "model_id": "Qwen3.5-2B-Q4_K_M.gguf",
        "example_id": "spider:db:0",
        "db_id": "s1",
        "predicted_sql": "SELECT 1",
        "retry_count": 0,
        "execution_success": True,
        "definitive_failure": False,
        "error": None,
        "raw_output": "SELECT 1",
        "input_tokens": 80,
        "output_tokens": 5,
        "time_to_first_token_s": 0.5,
        "generation_time_s": 1.5,
        "query_execution_time_s": 0.1,
        "total_time_s": 1.6,
        "scores": {
            "normalized_exact_match": 0.8,
            "bertscore_f": 0.9,
            "rouge_l_f": 0.7,
        },
        "attempts": [
            {
                "llm_metrics": {
                    "time_to_first_token_s": 0.5,
                    "total_generation_time_s": 1.5,
                }
            }
        ],
    }
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in (direct_result, sql_result)),
        encoding="utf-8",
    )
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"

    return_code = main(
        [
            "--results",
            str(results_path),
            "--serialization-report",
            str(report_path),
            "--tables-dir",
            str(tables_dir),
            "--figures-dir",
            str(figures_dir),
            "--context-window",
            "250",
            "--allow-incomplete",
        ]
    )

    assert return_code == 0
    assert (tables_dir / EXPERIMENT_OUTCOMES_TABLE_FILENAME).is_file()
    assert (tables_dir / SERIALIZATION_EFFICIENCY_TABLE_FILENAME).is_file()
    assert (figures_dir / TOKENIZER_COMPARISON_FIGURE_FILENAME).stat().st_size > 0
    assert (figures_dir / DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME).stat().st_size > 0
    assert (figures_dir / DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME).stat().st_size > 0
    for metric in DIRECT_ACCURACY_METRIC_ORDER:
        assert (figures_dir / DIRECT_DATABASE_SIZE_FIGURE_FILENAMES[metric]).stat().st_size > 0
    assert (tables_dir / DIRECT_OPERATIONAL_TABLE_FILENAME).is_file()
    assert (figures_dir / DIRECT_TOKEN_USAGE_FIGURE_FILENAME).stat().st_size > 0
    assert (figures_dir / DIRECT_LATENCY_FIGURE_FILENAME).stat().st_size > 0
    assert (tables_dir / TEXT_TO_SQL_RETRIEVAL_TABLE_FILENAME).is_file()
    assert (figures_dir / TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME).stat().st_size > 0
    assert (tables_dir / TEXT_TO_SQL_OPERATIONAL_TABLE_FILENAME).is_file()
    assert (figures_dir / TEXT_TO_SQL_TOKEN_FIGURE_FILENAME).stat().st_size > 0
    assert (figures_dir / TEXT_TO_SQL_LATENCY_FIGURE_FILENAME).stat().st_size > 0
    assert (tables_dir / TEXT_TO_SQL_RETRY_TABLE_FILENAME).is_file()
    assert (figures_dir / TEXT_TO_SQL_RETRY_FIGURE_FILENAME).stat().st_size > 0

    figure_filenames = (
        TOKENIZER_COMPARISON_FIGURE_FILENAME,
        DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME,
        DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME,
        *DIRECT_DATABASE_SIZE_FIGURE_FILENAMES.values(),
        DIRECT_TOKEN_USAGE_FIGURE_FILENAME,
        DIRECT_LATENCY_FIGURE_FILENAME,
        TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME,
        TEXT_TO_SQL_TOKEN_FIGURE_FILENAME,
        TEXT_TO_SQL_LATENCY_FIGURE_FILENAME,
        TEXT_TO_SQL_RETRY_FIGURE_FILENAME,
    )
    for filename in figure_filenames:
        assert (figures_dir / filename).read_bytes().startswith(b"%PDF-")


def test_load_token_report_rejects_non_numeric_tokens(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    row = _record(
        dataset="spider",
        database_id="s1",
        serialization_format="csv",
        family="qwen-3.5",
        label="Qwen 3.5",
        tokens=1,
    )
    row["tokens"] = "not-a-number"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-numeric token count"):
        load_token_report(path)
