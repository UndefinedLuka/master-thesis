"""Tests for direct-context operational-efficiency calculations and artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd

from visualizations.config import (
    DIRECT_LATENCY_FIGURE_FILENAME,
    DIRECT_OPERATIONAL_TABLE_FILENAME,
    DIRECT_TOKEN_USAGE_FIGURE_FILENAME,
)
from visualizations.direct_efficiency_data import calculate_direct_efficiency
from visualizations.direct_latency_by_format import (
    generate_direct_latency_figure,
    plot_direct_latency,
)
from visualizations.direct_operational_efficiency_table import (
    generate_direct_operational_table,
    render_direct_operational_table,
)
from visualizations.direct_token_usage import (
    generate_direct_token_usage_figure,
    plot_direct_token_usage,
)


def _result(
    *,
    dataset: str = "spider",
    model: str = "Qwen3.5-2B-Q4.gguf",
    fmt: str = "csv",
    example_id: str = "spider:db:0",
    db_id: str = "db",
    input_tokens: int = 100,
    output_tokens: int = 10,
    ttft: float = 2.0,
    total: float = 5.0,
    error: str | None = None,
    has_answer: bool = True,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "db_id": db_id,
        "example_id": example_id,
        "model_id": model,
        "pipeline": "direct",
        "format": fmt,
        "error": error,
        "has_answer": has_answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "time_to_first_token_s": ttft,
        "generation_time_s": total,
        "total_time_s": total,
    }


def _complete_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("spider", "bird"):
        for model_index, model in enumerate(
            ("Qwen3.5-2B-Q4.gguf", "gemma-4-E4B-Q4.gguf"), start=1
        ):
            for format_index, fmt in enumerate(("csv", "json"), start=1):
                rows.append(
                    _result(
                        dataset=dataset,
                        model=model,
                        fmt=fmt,
                        example_id=f"{dataset}:db:0",
                        db_id=f"{dataset}-db",
                        input_tokens=100 * format_index,
                        output_tokens=10 * model_index + format_index,
                        ttft=float(format_index),
                        total=float(format_index + model_index + 2),
                    )
                )
    return pd.DataFrame(rows)


def test_matched_cohort_removes_incomplete_question_from_all_formats() -> None:
    rows = _complete_matrix().to_dict("records")
    # This second Spider question is missing one model/format combination and must be removed
    # entirely rather than making CSV average over a different workload than JSON.
    rows.extend(
        [
            _result(example_id="spider:db:1", fmt="csv", input_tokens=10),
            _result(example_id="spider:db:1", fmt="json", input_tokens=20),
            _result(
                example_id="spider:db:1",
                model="gemma-4-E4B-Q4.gguf",
                fmt="csv",
                input_tokens=30,
            ),
        ]
    )

    analysis = calculate_direct_efficiency(pd.DataFrame(rows))
    summary = analysis.summary.set_index(["dataset", "format"])

    assert analysis.question_counts == {"bird": 1, "spider": 1}
    assert analysis.database_counts == {"bird": 1, "spider": 1}
    assert analysis.model_count == 2
    assert analysis.observation_count == 8
    assert summary.loc[("spider", "csv"), "average_input_tokens"] == 100
    assert summary.loc[("spider", "json"), "average_input_tokens"] == 200


def test_operational_means_split_ttft_generation_and_total() -> None:
    results = pd.DataFrame(
        [
            _result(input_tokens=100, output_tokens=10, ttft=2.0, total=7.0),
            _result(
                model="gemma-4-E4B-Q4.gguf",
                input_tokens=300,
                output_tokens=30,
                ttft=4.0,
                total=9.0,
            ),
        ]
    )

    row = calculate_direct_efficiency(results).summary.iloc[0]

    assert row["average_input_tokens"] == 200
    assert row["average_output_tokens"] == 20
    assert row["average_ttft_s"] == pytest.approx(3.0)
    assert row["average_generation_time_s"] == pytest.approx(5.0)
    assert row["average_total_latency_s"] == pytest.approx(8.0)
    assert row["average_ttft_s"] + row["average_generation_time_s"] == pytest.approx(
        row["average_total_latency_s"]
    )


def test_operational_analysis_rejects_inconsistent_direct_timers() -> None:
    result = _result()
    result["generation_time_s"] = 4.0
    result["total_time_s"] = 5.0

    with pytest.raises(ValueError, match="different generation and total timers"):
        calculate_direct_efficiency(pd.DataFrame([result]))


def test_table_and_figures_contain_all_requested_metrics(tmp_path: Path) -> None:
    analysis = calculate_direct_efficiency(_complete_matrix())
    latex = render_direct_operational_table(analysis)

    assert "Input tokens" in latex
    assert "Output tokens" in latex
    assert "TTFT (s)" in latex
    assert "Generation (s)" in latex
    assert "Total latency (s)" in latex
    assert "Generation time is measured from the first output token" in latex

    token_figure, token_axes = plot_direct_token_usage(analysis)
    latency_figure, latency_axes = plot_direct_latency(analysis)
    assert [axis.get_title() for axis in token_axes[0]] == [
        "Average input tokens",
        "Average output tokens",
    ]
    assert len(latency_axes[0].containers) == 2
    assert [text.get_text() for text in latency_figure.legends[0].get_texts()] == [
        "Time to first token",
        "Generation after first token",
    ]

    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    table_path = generate_direct_operational_table(analysis, tables_dir)
    token_path = generate_direct_token_usage_figure(analysis, figures_dir)
    latency_path = generate_direct_latency_figure(analysis, figures_dir)
    assert table_path == tables_dir / DIRECT_OPERATIONAL_TABLE_FILENAME
    assert token_path == figures_dir / DIRECT_TOKEN_USAGE_FIGURE_FILENAME
    assert latency_path == figures_dir / DIRECT_LATENCY_FIGURE_FILENAME
    assert token_path.stat().st_size > 0
    assert latency_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(token_figure)
    plt.close(latency_figure)
