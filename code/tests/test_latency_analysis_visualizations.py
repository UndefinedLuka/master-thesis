"""Tests for latency-distribution and TTFT/input-token visualizations."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from visualizations.latency_distribution_by_format import (  # noqa: E402
    plot_latency_distribution,
    prepare_latency_data,
)
from visualizations.latency_distribution_by_format import (
    save_figure as save_latency_figure,
)
from visualizations.ttft_vs_input_tokens import (  # noqa: E402
    ATTEMPT_INPUT_TOKENS_COLUMN,
    ATTEMPT_TTFT_COLUMN,
    plot_ttft_vs_input_tokens,
    prepare_ttft_data,
)
from visualizations.ttft_vs_input_tokens import (
    save_figure as save_ttft_figure,
)


def test_prepare_latency_data_maps_sql_and_removes_invalid_rows() -> None:
    results = pd.DataFrame(
        {
            "dataset": ["bird", "bird", "spider", "spider"],
            "pipeline": ["direct", "text_to_sql", "direct", "direct"],
            "format": ["csv", None, "json", "xml"],
            "total_time_s": [10.0, 2.5, 0.0, "invalid"],
        }
    )

    data = prepare_latency_data(results)

    assert data["method"].astype("object").tolist() == ["csv", "SQL execution"]
    assert data["total_time_s"].tolist() == [10.0, 2.5]


def test_latency_distribution_has_dataset_facets_and_log_scale(tmp_path: Path) -> None:
    data = prepare_latency_data(
        pd.DataFrame(
            {
                "dataset": ["bird"] * 6 + ["spider"] * 6,
                "pipeline": ["direct"] * 12,
                "format": ["csv"] * 3 + ["xml"] * 3 + ["csv"] * 3 + ["xml"] * 3,
                "total_time_s": [1.0, 2.0, 4.0, 5.0, 10.0, 20.0] * 2,
            }
        )
    )

    figure, axes = plot_latency_distribution(data)
    output = tmp_path / "latency.png"
    save_latency_figure(figure, output)

    assert [axis.get_title() for axis in axes] == ["BIRD", "Spider"]
    assert all(axis.get_yscale() == "log" for axis in axes)
    assert axes[0].get_xlabel() == "Method"
    assert [label.get_text() for label in axes[0].get_xticklabels()] == ["CSV", "XML"]
    assert output.is_file()
    assert output.stat().st_size > 0


def test_prepare_ttft_data_pairs_sql_first_attempt_metrics() -> None:
    results = pd.DataFrame(
        {
            "dataset": ["bird", "bird", "spider", "spider"],
            "pipeline": ["direct", "text_to_sql", "direct", "direct"],
            "format": ["csv", None, "json", "xml"],
            "input_tokens": [10_000, 3_000, 0, 500],
            "time_to_first_token_s": [2.0, 0.2, 1.0, 0.0],
            ATTEMPT_INPUT_TOKENS_COLUMN: [None, [1_000, 2_000], None, None],
            ATTEMPT_TTFT_COLUMN: [None, [0.2, 0.3], None, None],
        }
    )

    data = prepare_ttft_data(results)

    assert data["method"].astype("object").tolist() == ["csv", "SQL execution"]
    sql = data[data["method"].eq("SQL execution")].iloc[0]
    assert sql["input_tokens"] == 1_000
    assert sql["time_to_first_token_s"] == pytest.approx(0.2)


def test_ttft_plot_combines_datasets_with_log_axes_and_legend(tmp_path: Path) -> None:
    data = prepare_ttft_data(
        pd.DataFrame(
            {
                "dataset": ["bird"] * 6 + ["spider"] * 6,
                "pipeline": ["direct"] * 12,
                "format": ["csv"] * 3 + ["xml"] * 3 + ["csv"] * 3 + ["xml"] * 3,
                "input_tokens": [100, 200, 400, 200, 400, 800] * 2,
                "time_to_first_token_s": [0.1, 0.2, 0.4, 0.2, 0.4, 0.8] * 2,
            }
        )
    )

    figure, axis = plot_ttft_vs_input_tokens(data)
    output = tmp_path / "ttft.png"
    save_ttft_figure(figure, output)

    assert len(figure.axes) == 1
    assert axis.get_title() == ""
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert axis.get_xlabel() == "Input tokens"
    assert axis.get_ylabel() == "Time to First Token (seconds)"
    assert any(
        "BIRD and Spider observations are combined" in text.get_text()
        for text in figure.texts
    )
    assert figure.legends[0].get_title().get_text() == "Method"
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["CSV", "XML"]
    assert output.is_file()
    assert output.stat().st_size > 0
