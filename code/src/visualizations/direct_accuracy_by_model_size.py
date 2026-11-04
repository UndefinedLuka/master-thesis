"""Produce direct-context retrieval-accuracy scaling curves by model family and size."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    DIRECT_ACCURACY_METRIC_ORDER,
    DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME,
    METRIC_TITLES,
    TOKENIZER_FAMILY_COLORS,
    TOKENIZER_FAMILY_LABELS,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.direct_accuracy_data import calculate_model_size_accuracy
from visualizations.plot_numbers import number, number_formatter, save_pgf_figure, tight_layout
from visualizations.uncertainty import interval_marks

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_direct_accuracy_by_model_size(
    summary: pd.DataFrame,
) -> tuple[Figure, list[list[Axes]]]:
    """Plot family scaling curves with metric rows and dataset columns."""
    required = {
        "dataset",
        "model_id",
        "model_family",
        "model_size_b",
        "metric",
        "mean_score",
        "format_count",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Model-size accuracy summary is missing: {', '.join(sorted(missing))}")
    if summary.empty:
        raise ValueError("No direct-context model-size accuracy values are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    datasets = [value for value in DATASET_ORDER if value in set(summary["dataset"])]
    metrics = [
        value for value in DIRECT_ACCURACY_METRIC_ORDER if value in set(summary["metric"])
    ]
    families = [
        value for value in TOKENIZER_FAMILY_ORDER if value in set(summary["model_family"])
    ]
    metric_limits = {
        metric: _score_limits(summary[summary["metric"].eq(metric)]["mean_score"])
        for metric in metrics
    }
    figure, axes_array = plt.subplots(
        nrows=len(metrics),
        ncols=len(datasets),
        figsize=(5.4 * len(datasets), 4.2 * len(metrics)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    axes = [list(row) for row in axes_array]

    all_sizes = sorted(float(value) for value in summary["model_size_b"].unique())
    for column_index, dataset in enumerate(datasets):
        dataset_rows = summary[summary["dataset"].eq(dataset)]
        for row_index, metric in enumerate(metrics):
            axis = axes[row_index][column_index]
            panel = dataset_rows[dataset_rows["metric"].eq(metric)]
            for family in families:
                family_rows = panel[panel["model_family"].eq(family)].sort_values(
                    ["model_size_b", "model_id"]
                )
                if family_rows.empty:
                    continue
                axis.plot(
                    family_rows["model_size_b"],
                    family_rows["mean_score"],
                    marker="o",
                    markersize=6,
                    linewidth=2,
                    color=TOKENIZER_FAMILY_COLORS[family],
                    label=TOKENIZER_FAMILY_LABELS[family],
                )
                interval_marks(axis, family_rows["model_size_b"], family_rows, "mean_score",
                               color=TOKENIZER_FAMILY_COLORS[family])
            axis.set_xscale("log")
            axis.set_xticks(all_sizes, [_format_size(value) for value in all_sizes])
            axis.minorticks_off()
            axis.yaxis.set_major_formatter(number_formatter())
            axis.axhline(0.0, color="#444444", linewidth=0.8)
            axis.grid(axis="both", color="#D0D0D0", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0:
                axis.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{METRIC_TITLES.get(metric, metric)}\nMean score")
            if row_index == len(metrics) - 1:
                axis.tick_params(axis="x", labelrotation=60)

            low, high = metric_limits[metric]
            if "mean_score_high" in summary:
                all_metric = summary[summary["metric"].eq(metric)]
                low = min(low, float(all_metric["mean_score_low"].min()))
                high = max(high, float(all_metric["mean_score_high"].max()) * 1.08)
            axis.set_ylim(low, high)

    figure.supxlabel(
        "Nominal model size (billions of parameters, log scale)",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TOKENIZER_FAMILY_COLORS[family],
            marker="o",
            linewidth=2,
            label=TOKENIZER_FAMILY_LABELS[family],
        )
        for family in families
    ]
    figure.legend(
        handles=legend_handles,
        title="Model family",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=max(1, len(families)),
        frameon=False,
    )
    tight_layout(figure, rect=(0.03, 0, 1.0, 0.94))
    return figure, axes


def generate_direct_accuracy_by_model_size_figure(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> Path:
    """Calculate, plot, and save the direct model-size accuracy figure."""
    output_path = output_dir / DIRECT_ACCURACY_MODEL_SIZE_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_direct_accuracy_by_model_size(
        calculate_model_size_accuracy(
            results,
            excluded_model_markers=excluded_model_markers,
        )
    )

    import matplotlib.pyplot as plt

    save_pgf_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _score_limits(values: pd.Series) -> tuple[float, float]:
    numeric = [float(value) for value in values if not pd.isna(value)]
    lower = min(0.0, min(numeric, default=0.0))
    upper = max(numeric, default=1.0)
    margin = max(0.03, (upper - lower) * 0.15)
    return lower - (0.02 if lower < 0 else 0.0), upper + margin


def _format_size(size: float) -> str:
    return number(size)
