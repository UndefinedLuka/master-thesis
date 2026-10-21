"""Produce Text-to-SQL retrieval-accuracy scaling curves by model family and size."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    DIRECT_ACCURACY_METRIC_ORDER,
    METRIC_TITLES,
    TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME,
    TOKENIZER_FAMILY_COLORS,
    TOKENIZER_FAMILY_LABELS,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.plot_numbers import number, number_formatter, save_pgf_figure, tight_layout
from visualizations.text_to_sql_data import TextToSqlAnalysis
from visualizations.uncertainty import interval_marks

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_text_to_sql_accuracy(
    analysis: TextToSqlAnalysis,
) -> tuple[Figure, list[list[Axes]]]:
    """Plot Text-to-SQL metric means against model size for each dataset and family."""
    summary = analysis.model_summary
    if summary.empty:
        raise ValueError("No Text-to-SQL model accuracy values are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes_array = plt.subplots(
        nrows=len(DATASET_ORDER),
        ncols=len(DIRECT_ACCURACY_METRIC_ORDER),
        figsize=(16.2, 8.4),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    axes = [list(row) for row in axes_array]
    all_sizes = sorted(float(value) for value in summary["model_size_b"].unique())
    metric_limits = {
        metric: _score_limits(summary[metric]) for metric in DIRECT_ACCURACY_METRIC_ORDER
    }

    for row_index, dataset in enumerate(DATASET_ORDER):
        dataset_rows = summary[summary["dataset"].eq(dataset)]
        for column_index, metric in enumerate(DIRECT_ACCURACY_METRIC_ORDER):
            axis = axes[row_index][column_index]
            for family in TOKENIZER_FAMILY_ORDER:
                family_rows = dataset_rows[
                    dataset_rows["model_family"].eq(family)
                ].sort_values(["model_size_b", "model_id"])
                if family_rows.empty:
                    continue
                axis.plot(
                    family_rows["model_size_b"],
                    family_rows[metric],
                    marker="o",
                    markersize=6,
                    linewidth=2,
                    color=TOKENIZER_FAMILY_COLORS[family],
                )
                interval_marks(axis, family_rows["model_size_b"], family_rows, metric,
                               color=TOKENIZER_FAMILY_COLORS[family])
            axis.set_xscale("log")
            axis.set_xticks(all_sizes, [_format_size(value) for value in all_sizes])
            axis.minorticks_off()
            axis.yaxis.set_major_formatter(number_formatter())
            axis.grid(axis="both", color="#D0D0D0", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(METRIC_TITLES[metric], fontweight="bold")
            low, high = metric_limits[metric]
            if f"{metric}_high" in summary:
                low = min(low, float(summary[f"{metric}_low"].min()))
                high = max(high, float(summary[f"{metric}_high"].max()) * 1.08)
            axis.set_ylim(low, high)
            if column_index == 0:
                axis.set_ylabel(f"{DATASET_LABELS[dataset]}\nMean score")
            if row_index == len(DATASET_ORDER) - 1:
                axis.set_xlabel("Nominal model size (billions, log scale)")
                axis.tick_params(axis="x", labelrotation=35, labelsize=8.5)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linewidth=2,
            color=TOKENIZER_FAMILY_COLORS[family],
            label=TOKENIZER_FAMILY_LABELS[family],
        )
        for family in TOKENIZER_FAMILY_ORDER
    ]
    figure.legend(
        handles=handles,
        title="Model family",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=False,
    )
    tight_layout(figure, rect=(0.03, 0, 1.0, 0.94))
    return figure, axes


def generate_text_to_sql_accuracy_figure(
    analysis: TextToSqlAnalysis,
    output_dir: Path,
) -> Path:
    """Plot and save Text-to-SQL retrieval accuracy by model size."""
    output_path = output_dir / TEXT_TO_SQL_ACCURACY_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_text_to_sql_accuracy(analysis)

    import matplotlib.pyplot as plt

    save_pgf_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _score_limits(values: pd.Series) -> tuple[float, float]:
    numeric = [float(value) for value in values if not pd.isna(value)]
    lower = min(0.0, min(numeric, default=0.0))
    upper = max(numeric, default=1.0)
    margin = max(0.03, (upper - lower) * 0.12)
    return lower, upper + margin


def _format_size(size: float) -> str:
    return number(size)
