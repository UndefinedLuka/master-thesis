"""Produce a matched input/output-token comparison, without conflating their costs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from visualizations.common_cohort_data import CommonCohortAnalysis
from visualizations.common_cohort_plotting import (
    cohort_datasets,
    cohort_methods,
    panel_title,
    save_figure,
    style_axis,
)
from visualizations.config import (
    COMMON_COHORT_METHOD_LABELS,
    COMMON_COHORT_SQL_METHOD,
    COMMON_COHORT_TOKENS_FIGURE_FILENAME,
    FORMAT_COLORS,
)
from visualizations.plot_numbers import number, number_formatter, tight_layout
from visualizations.uncertainty import interval_marks

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_common_cohort_token_usage(analysis: CommonCohortAnalysis) -> Figure:
    """Use log-scaled input bars and linear output bars, with absolute count labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    datasets = cohort_datasets(analysis)
    figure, axes = plt.subplots(2, len(datasets), figsize=(12.2, 8.2), squeeze=False)
    colors = {COMMON_COHORT_SQL_METHOD: "#374151", **FORMAT_COLORS}
    for column_index, dataset in enumerate(datasets):
        methods = cohort_methods(analysis, dataset)
        panel = analysis.summary[analysis.summary["dataset"].eq(dataset)].set_index("method")
        positions = list(range(len(methods)))
        for row_index, metric in enumerate(("average_input_tokens", "average_output_tokens")):
            axis = axes[row_index, column_index]
            values = [float(panel.loc[method, metric]) for method in methods]
            axis.barh(positions, values, color=[colors[m] for m in methods], height=0.64)
            interval_marks(axis, positions, panel.reindex(methods), metric, horizontal=True)
            upper = (panel.reindex(methods)[f"{metric}_high"].tolist()
                     if f"{metric}_high" in panel else values)
            upper = [bound if np.isfinite(bound) else value
                     for bound, value in zip(upper, values, strict=True)]
            for position, method, value in zip(positions, methods, values, strict=True):
                label = number(value, 0 if row_index == 0 else 1)
                end = upper[position]
                x = end * 1.08 if row_index == 0 else end + max(upper) * 0.025
                axis.text(
                    x,
                    position,
                    label,
                    va="center",
                    fontweight="bold" if method == COMMON_COHORT_SQL_METHOD else "normal",
                )
            axis.axhspan(-0.45, 0.45, color="#EEF2F5", zorder=0)
            axis.axhline(0.5, color="#AAB2B9", linewidth=0.8)
            axis.set_yticks(positions, [COMMON_COHORT_METHOD_LABELS[m] for m in methods])
            axis.invert_yaxis()
            if row_index == 0:
                axis.set_xscale("log")
                axis.set_xlim(max(1, min(values) / 2), max(upper) * 5.5)
                axis.xaxis.set_major_formatter(
                    FuncFormatter(
                        lambda value, _: number(value / 1000) + r"\,k"
                        if value >= 1000
                        else number(value)
                    )
                )
                axis.xaxis.set_minor_formatter(NullFormatter())
                axis.set_xlabel("Mean input tokens (log scale)")
                axis.set_title(
                    panel_title(analysis, dataset), fontweight="bold", pad=14
                )
            else:
                axis.set_xlim(0, max(upper) * 1.25)
                axis.xaxis.set_major_formatter(number_formatter())
                axis.set_xlabel("Mean output tokens (linear scale)")
            style_axis(axis)
    tight_layout(figure, rect=(0, 0, 1, 1), w_pad=2.5, h_pad=2.2)
    return figure


def generate_common_cohort_token_figure(
    analysis: CommonCohortAnalysis,
    output_dir: Path,
) -> Path:
    return save_figure(
        plot_common_cohort_token_usage(analysis), output_dir, COMMON_COHORT_TOKENS_FIGURE_FILENAME
    )
