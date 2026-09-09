"""Produce the direct-context retrieval-accuracy comparison by serialization format."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME,
    DIRECT_ACCURACY_METRIC_ORDER,
    FORMAT_COLORS,
    FORMAT_LABELS,
    METRIC_TITLES,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.direct_accuracy_data import calculate_format_accuracy

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_direct_accuracy_by_format(
    summary: pd.DataFrame,
) -> tuple[Figure, list[list[Axes]]]:
    """Plot mean scores in a dataset-by-metric grid with one bar per format."""
    required = {
        "dataset",
        "format",
        "metric",
        "mean_score",
        "observation_count",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Format accuracy summary is missing: {', '.join(sorted(missing))}")
    if summary.empty:
        raise ValueError("No direct-context format accuracy values are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [value for value in DATASET_ORDER if value in set(summary["dataset"])]
    metrics = [
        value for value in DIRECT_ACCURACY_METRIC_ORDER if value in set(summary["metric"])
    ]
    formats = [
        value for value in SERIALIZATION_FORMAT_ORDER if value in set(summary["format"])
    ]
    metric_limits = {
        metric: _score_limits(summary[summary["metric"].eq(metric)]["mean_score"])
        for metric in metrics
    }
    figure, axes_array = plt.subplots(
        nrows=len(datasets),
        ncols=len(metrics),
        figsize=(5.4 * len(metrics), 4.2 * len(datasets)),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    axes = [list(row) for row in axes_array]

    for row_index, dataset in enumerate(datasets):
        dataset_rows = summary[summary["dataset"].eq(dataset)]
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index][column_index]
            panel = dataset_rows[dataset_rows["metric"].eq(metric)].set_index("format")
            scores = [
                float(panel.loc[value, "mean_score"]) if value in panel.index else float("nan")
                for value in formats
            ]
            bars = axis.bar(
                range(len(formats)),
                scores,
                color=[FORMAT_COLORS[value] for value in formats],
                width=0.72,
            )
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            axis.axhline(0.0, color="#444444", linewidth=0.8)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(METRIC_TITLES.get(metric, metric), fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{DATASET_LABELS.get(dataset, dataset)}\nMean score")
            if row_index == len(datasets) - 1:
                axis.set_xticks(
                    range(len(formats)),
                    [FORMAT_LABELS[value] for value in formats],
                    rotation=30,
                    ha="right",
                )
            axis.set_ylim(*metric_limits[metric])

    figure.suptitle(
        "Direct-context retrieval accuracy by serialization format",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            "Bars are arithmetic means across all retained models and questions; "
            "zero-scored failed runs remain included."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.03, 0.05, 1.0, 0.96))
    return figure, axes


def generate_direct_accuracy_by_format_figure(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> Path:
    """Calculate, plot, and save the direct-format accuracy figure."""
    output_path = output_dir / DIRECT_ACCURACY_FORMAT_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_direct_accuracy_by_format(
        calculate_format_accuracy(
            results,
            excluded_model_markers=excluded_model_markers,
        )
    )

    import matplotlib.pyplot as plt

    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _score_limits(values: pd.Series) -> tuple[float, float]:
    numeric = [float(value) for value in values if not pd.isna(value)]
    lower = min(0.0, min(numeric, default=0.0))
    upper = max(numeric, default=1.0)
    margin = max(0.03, (upper - lower) * 0.20)
    return lower - (0.02 if lower < 0 else 0.0), upper + margin
