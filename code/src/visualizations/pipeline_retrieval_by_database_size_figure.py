"""Plot both pipelines' answerable-cohort accuracy within fixed database-size tiers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from visualizations.config import (
    COMMON_COHORT_METHOD_ORDER,
    COMMON_COHORT_SQL_METHOD,
    DIRECT_ACCURACY_METRIC_ORDER,
    FORMAT_COLORS,
    FORMAT_LABELS,
    METRIC_TITLES,
)
from visualizations.pipeline_retrieval_by_database_size import (
    SIZE_TIER_LABELS,
    SIZE_TIER_ORDER,
    PipelineSizeRetrievalAnalysis,
)
from visualizations.plot_numbers import number_formatter, save_pgf_figure, tight_layout
from visualizations.uncertainty import interval_marks

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PIPELINE_SIZE_RETRIEVAL_FIGURE_FILENAME = "pipeline_retrieval_by_database_size.pdf"


def plot_pipeline_size_retrieval_comparison(analysis: PipelineSizeRetrievalAnalysis) -> Figure:
    """Show pooled size-tier accuracy in two top panels and one centered bottom panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summary = analysis.summary
    if summary.empty:
        raise ValueError("No size-tier retrieval comparisons are available to plot")
    methods = COMMON_COHORT_METHOD_ORDER
    colors = {COMMON_COHORT_SQL_METHOD: "#222222", **FORMAT_COLORS}
    offsets = dict(zip(methods, np.linspace(-0.27, 0.27, len(methods)), strict=True))
    figure = plt.figure(figsize=(7.2, 7.2))
    grid = figure.add_gridspec(2, 4)
    axes = [figure.add_subplot(grid[0, :2])]
    axes.extend(
        figure.add_subplot(position, sharex=axes[0], sharey=axes[0])
        for position in (grid[0, 2:], grid[1, 1:3])
    )
    axes[1].tick_params(axis="y", labelleft=False)
    for index, (axis, metric) in enumerate(zip(axes, DIRECT_ACCURACY_METRIC_ORDER, strict=True)):
        panel = summary.loc[summary["metric"].eq(metric)]
        for position, tier in enumerate(SIZE_TIER_ORDER):
            tier_rows = panel.loc[panel["size_tier"].eq(tier)].set_index("method")
            if tier_rows.empty:
                axis.axvspan(position - 0.4, position + 0.4, color="#f2f2f2", zorder=0)
                axis.text(position, 0.5, "No databases", ha="center", va="center",
                          rotation=90, color="#777777")
                continue
            for method in methods:
                point = position + offsets[method]
                sql = method == COMMON_COHORT_SQL_METHOD
                interval_marks(axis, [point], tier_rows.loc[[method]], "mean_score",
                               color=colors[method])
                axis.plot(point, tier_rows.loc[method, "mean_score"],
                          marker="D" if sql else "o", linestyle="none",
                          color=colors[method], markersize=5 if sql else 3.7,
                          zorder=7 if sql else 6)
        lower = min(0.0, float(summary[["mean_score", "mean_score_low"]].min().min()))
        upper = max(1.0, float(summary[["mean_score", "mean_score_high"]].max().max()))
        axis.set_ylim(lower - 0.035, upper + 0.035)
        axis.set_xlim(-0.5, len(SIZE_TIER_ORDER) - 0.5)
        axis.set_xticks(range(len(SIZE_TIER_ORDER)),
                        [SIZE_TIER_LABELS[tier] for tier in SIZE_TIER_ORDER])
        axis.set_yticks(np.linspace(0, 1, 5))
        axis.yaxis.set_major_formatter(number_formatter())
        axis.tick_params(axis="x", labelbottom=True, length=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#aaaaaa")
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e8e8e8", linewidth=0.6)
        axis.set_title(METRIC_TITLES[metric], pad=9)
        if index != 1:
            axis.set_ylabel("Mean score")
    handles = [
        Line2D([], [], color=colors[method], marker="D" if method == COMMON_COHORT_SQL_METHOD
               else "o", linestyle="none", markersize=5,
               label="Text-to-SQL" if method == COMMON_COHORT_SQL_METHOD else FORMAT_LABELS[method])
        for method in methods
    ]
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                  handletextpad=0.4, columnspacing=1.3)
    figure.supxlabel("Database-size tier (mean serialised payload tokens)")
    tight_layout(figure, rect=(0, 0.025, 1, 0.88), h_pad=2.0, w_pad=1.2)
    return figure


def generate_pipeline_size_retrieval_figure(
    analysis: PipelineSizeRetrievalAnalysis, figures_dir: Path,
) -> Path:
    """Write the tier-stratified retrieval PDF from an already validated analysis."""
    import matplotlib.pyplot as plt

    path = figures_dir / PIPELINE_SIZE_RETRIEVAL_FIGURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_pipeline_size_retrieval_comparison(analysis)
    try:
        save_pgf_figure(figure, path)
    finally:
        plt.close(figure)
    return path
