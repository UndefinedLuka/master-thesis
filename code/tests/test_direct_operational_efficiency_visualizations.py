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
    SERIALIZATION_FORMAT_ORDER,
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
            for format_index, fmt in enumerate(SERIALIZATION_FORMAT_ORDER, start=1):
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


@pytest.mark.parametrize("missing", ["run", "format", "model_question"])
def test_incomplete_selected_matrix_raises_instead_of_changing_cohort(missing: str) -> None:
    data = _complete_matrix()
    if missing == "run":
        data = data.drop(data.index[0])
    elif missing == "format":
        data = data.loc[~data["format"].eq("json")]
    else:
        data = data.loc[~(data["dataset"].eq("bird") & data["model_id"].str.startswith("Qwen"))]
    with pytest.raises(ValueError, match="identical question/model sets"):
        calculate_direct_efficiency(data)


def test_blank_outputs_and_recorded_failures_retain_costs() -> None:
    data = _complete_matrix()
    data["has_answer"] = False
    data["error"] = "recorded output failure"
    analysis = calculate_direct_efficiency(data)
    assert analysis.question_counts == {"bird": 1, "spider": 1}
    assert analysis.observation_count == len(data)
    assert analysis.model_count == 2
    expected = data.groupby(["dataset", "format"])["input_tokens"].mean()
    actual = analysis.summary.set_index(["dataset", "format"])["average_input_tokens"]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


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

    results = pd.concat([results.assign(format=fmt) for fmt in SERIALIZATION_FORMAT_ORDER])
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
        calculate_direct_efficiency(pd.DataFrame([dict(result, format=fmt)
                                                for fmt in SERIALIZATION_FORMAT_ORDER]))


@pytest.mark.parametrize("column,value", [
    ("input_tokens", float("inf")), ("output_tokens", float("nan")),
    ("output_tokens", -1), ("input_tokens", 1.5), ("output_tokens", True),
    ("input_tokens", 0), ("time_to_first_token_s", 100),
])
def test_invalid_recorded_measurements_raise(column: str, value: object) -> None:
    data = _complete_matrix().astype({column: object})
    data.at[0, column] = value
    with pytest.raises(ValueError):
        calculate_direct_efficiency(data)


def test_duplicate_direct_identity_raises() -> None:
    data = _complete_matrix()
    with pytest.raises(ValueError, match="Duplicate operational run"):
        calculate_direct_efficiency(pd.concat([data, data.iloc[[0]]]))


def test_table_contains_all_requested_metrics(tmp_path: Path) -> None:
    analysis = calculate_direct_efficiency(_complete_matrix())
    latex = render_direct_operational_table(analysis)

    assert "Input tokens" in latex
    assert "Output tokens" in latex
    assert "TTFT (s)" in latex
    assert "Generation (s)" in latex
    assert "Total latency (s)" in latex
    assert r"\parbox" not in latex
    assert "Note:" not in latex

    tables_dir = tmp_path / "tables"
    table_path = generate_direct_operational_table(analysis, tables_dir)
    assert table_path == tables_dir / DIRECT_OPERATIONAL_TABLE_FILENAME


def test_figures_contain_all_requested_metrics(tmp_path: Path) -> None:
    analysis = calculate_direct_efficiency(_complete_matrix())
    token_figure, token_axes = plot_direct_token_usage(analysis)
    latency_figure, latency_axes = plot_direct_latency(analysis)
    assert [axis.get_title() for axis in token_axes[0]] == [
        "Average input tokens",
        "Average output tokens",
    ]
    assert len(latency_axes[0].containers) == 2
    assert [text.get_text() for text in latency_figure.legends[0].get_texts()] == [
        "TTFT",
        "Generation after first token",
    ]

    figures_dir = tmp_path / "figures"
    token_path = generate_direct_token_usage_figure(analysis, figures_dir)
    latency_path = generate_direct_latency_figure(analysis, figures_dir)
    assert token_path == figures_dir / DIRECT_TOKEN_USAGE_FIGURE_FILENAME
    assert latency_path == figures_dir / DIRECT_LATENCY_FIGURE_FILENAME
    assert token_path.stat().st_size > 0
    assert latency_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(token_figure)
    plt.close(latency_figure)
