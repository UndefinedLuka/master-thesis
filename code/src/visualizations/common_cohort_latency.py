"""Produce a matched latency decomposition for both extraction pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualizations.common_cohort_data import CommonCohortAnalysis
from visualizations.common_cohort_plotting import (
    cohort_datasets,
    cohort_methods,
    panel_title,
    save_figure,
    style_axis,
)
from visualizations.config import (
    COMMON_COHORT_LATENCY_FIGURE_FILENAME,
    COMMON_COHORT_METHOD_LABELS,
    COMMON_COHORT_SQL_METHOD,
    TEXT_TO_SQL_LATENCY_PHASE_COLORS,
)
from visualizations.uncertainty import interval_marks

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_common_cohort_latency(analysis: CommonCohortAnalysis) -> Figure:
    """Show absolute per-run means, with SQL as the common baseline in each dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = cohort_datasets(analysis)
    figure, axes = plt.subplots(1, len(datasets), figsize=(12.2, 5.1), squeeze=False)
    phases = (
        ("average_ttft_s", "ttft", "TTFT (cumulative for SQL)"),
        ("average_generation_time_s", "generation", "Generation after first tokens"),
        ("average_sql_execution_time_s", "sql", "SQL execution"),
    )
    for axis, dataset in zip(axes[0], datasets, strict=True):
        methods = cohort_methods(analysis, dataset)
        panel = analysis.summary[analysis.summary["dataset"].eq(dataset)].set_index("method")
        positions = list(range(len(methods)))
        left = [0.0] * len(methods)
        for column, phase, label in phases:
            values = [float(panel.loc[method, column]) for method in methods]
            axis.barh(
                positions,
                values,
                left=left,
                height=0.64,
                color=TEXT_TO_SQL_LATENCY_PHASE_COLORS[phase],
                label=label,
            )
            left = [a + b for a, b in zip(left, values, strict=True)]
        maximum = max(left)
        interval_marks(axis, positions, panel.reindex(methods), "average_total_latency_s",
                       horizontal=True)
        if "average_total_latency_s_high" in panel:
            maximum = max(maximum, float(panel["average_total_latency_s_high"].max()))
        for position, method, total in zip(positions, methods, left, strict=True):
            axis.text(
                (float(panel.loc[method, "average_total_latency_s_high"])
                 if "average_total_latency_s_high" in panel else total) + maximum * 0.02,
                position,
                f"{total:.3f} s",
                va="center",
                fontsize=8.5,
                fontweight="bold" if method == COMMON_COHORT_SQL_METHOD else "normal",
            )
        axis.axhspan(-0.45, 0.45, color="#EEF2F5", zorder=0)
        axis.axhline(0.5, color="#AAB2B9", linewidth=0.8)
        axis.set_yticks(positions, [COMMON_COHORT_METHOD_LABELS[m] for m in methods])
        axis.invert_yaxis()
        axis.set_xlim(0, maximum * 1.44)
        axis.set_title(panel_title(analysis, dataset), fontsize=11, fontweight="bold", pad=14)
        axis.set_xlabel("Mean total latency (seconds)", fontsize=10)
        style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
    return figure


def generate_common_cohort_latency_figure(
    analysis: CommonCohortAnalysis,
    output_dir: Path,
) -> Path:
    return save_figure(
        plot_common_cohort_latency(analysis), output_dir, COMMON_COHORT_LATENCY_FIGURE_FILENAME
    )
