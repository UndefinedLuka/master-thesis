"""Produce the Text-to-SQL error-correction outcome and overhead figure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualizations.config import (
    DATASET_COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    RETRY_OUTCOME_LABELS,
    RETRY_OUTCOME_ORDER,
    TEXT_TO_SQL_RETRY_FIGURE_FILENAME,
)
from visualizations.text_to_sql_retry_data import TextToSqlRetryAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_error_correction_performance(
    analysis: TextToSqlRetryAnalysis,
) -> tuple[Figure, list[Axes]]:
    """Plot terminal attempt distributions and aggregate generation overhead by retry stage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    figure, axes_array = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15.5, 6.3),
        squeeze=False,
    )
    axes = list(axes_array[0])

    outcome_axis = axes[0]
    x_positions = list(range(len(RETRY_OUTCOME_ORDER)))
    width = 0.36
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        rows = analysis.outcome_summary[
            analysis.outcome_summary["dataset"].eq(dataset)
        ].set_index("terminal_outcome")
        values = [int(rows.loc[outcome, "run_count"]) for outcome in RETRY_OUTCOME_ORDER]
        positions = [value + (dataset_index - 0.5) * width for value in x_positions]
        bars = outcome_axis.bar(
            positions,
            values,
            width=width,
            color=DATASET_COLORS[dataset],
            label=DATASET_LABELS[dataset],
        )
        outcome_axis.bar_label(bars, labels=[f"{value:,}" for value in values], fontsize=8)
    outcome_axis.set_xticks(
        x_positions,
        [RETRY_OUTCOME_LABELS[outcome] for outcome in RETRY_OUTCOME_ORDER],
        rotation=24,
        ha="right",
    )
    outcome_axis.set_title("Terminal query-execution outcome", fontweight="bold")
    outcome_axis.set_ylabel("Runs")
    outcome_axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    outcome_axis.legend(title="Dataset", frameon=False)

    overhead_axis = axes[1]
    retry_positions = [0, 1, 2]
    retry_width = 0.36
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        rows = analysis.retry_stage_summary[
            analysis.retry_stage_summary["dataset"].eq(dataset)
        ].set_index("retry_index")
        values = [float(rows.loc[index, "total_generation_s"]) for index in (1, 2, 3)]
        positions = [value + (dataset_index - 0.5) * retry_width for value in retry_positions]
        bars = overhead_axis.bar(
            positions,
            values,
            width=retry_width,
            color=DATASET_COLORS[dataset],
            label=DATASET_LABELS[dataset],
        )
        overhead_axis.bar_label(
            bars,
            labels=[f"{value / 60:.1f} min" for value in values],
            fontsize=8,
        )
    overhead_axis.set_xticks(retry_positions, ["Retry 1", "Retry 2", "Retry 3"])
    overhead_axis.set_title("Additional model time by retry stage", fontweight="bold")
    overhead_axis.set_ylabel("Aggregate generation time (seconds)")
    overhead_axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    for axis in axes:
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout(rect=(0.03, 0, 1.0, 1.0))
    return figure, axes


def generate_error_correction_figure(
    analysis: TextToSqlRetryAnalysis,
    output_dir: Path,
) -> Path:
    """Plot and save the Text-to-SQL error-correction figure."""
    output_path = output_dir / TEXT_TO_SQL_RETRY_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_error_correction_performance(analysis)

    import matplotlib.pyplot as plt

    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path
