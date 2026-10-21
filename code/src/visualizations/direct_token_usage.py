"""Produce the direct-context average input/output token figure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    DIRECT_TOKEN_USAGE_FIGURE_FILENAME,
    FORMAT_COLORS,
    FORMAT_LABELS,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.direct_efficiency_data import DirectEfficiencyAnalysis
from visualizations.plot_numbers import number_formatter, save_pgf_figure, tight_layout
from visualizations.uncertainty import interval_marks

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

TOKEN_METRICS: tuple[tuple[str, str, str], ...] = (
    ("average_input_tokens", "Average input tokens", ",.0f"),
    ("average_output_tokens", "Average output tokens", ",.1f"),
)


def plot_direct_token_usage(
    analysis: DirectEfficiencyAnalysis,
) -> tuple[Figure, list[list[Axes]]]:
    """Plot average input and output token use in a dataset-by-token-type grid."""
    summary = analysis.summary
    required = {"dataset", "format", *(value[0] for value in TOKEN_METRICS)}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Token summary is missing: {', '.join(sorted(missing))}")
    if summary.empty:
        raise ValueError("No direct token measurements are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = [value for value in DATASET_ORDER if value in set(summary["dataset"])]
    formats = [
        value for value in SERIALIZATION_FORMAT_ORDER if value in set(summary["format"])
    ]
    figure, axes_array = plt.subplots(
        nrows=len(datasets),
        ncols=len(TOKEN_METRICS),
        figsize=(7.2 * len(TOKEN_METRICS), 4.2 * len(datasets)),
        sharex=True,
        squeeze=False,
    )
    axes = [list(row) for row in axes_array]

    for row_index, dataset in enumerate(datasets):
        panel = summary[summary["dataset"].eq(dataset)].set_index("format")
        for column_index, (metric, title, _number_format) in enumerate(TOKEN_METRICS):
            axis = axes[row_index][column_index]
            values = [
                float(panel.loc[value, metric]) if value in panel.index else float("nan")
                for value in formats
            ]
            axis.bar(
                range(len(formats)),
                values,
                color=[FORMAT_COLORS[value] for value in formats],
                width=0.72,
            )
            interval_marks(axis, range(len(formats)), panel.reindex(formats), metric)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(title, fontweight="bold")
            axis.yaxis.set_major_formatter(number_formatter())
            if column_index == 0:
                axis.set_ylabel(DATASET_LABELS.get(dataset, dataset))
            if row_index == len(datasets) - 1:
                axis.set_xticks(
                    range(len(formats)),
                    [FORMAT_LABELS[value] for value in formats],
                    rotation=30,
                    ha="right",
                )
            maximum = max((value for value in values if not pd.isna(value)), default=1.0)
            if f"{metric}_high" in panel:
                maximum = max(maximum, float(panel[f"{metric}_high"].max()))
            axis.set_ylim(top=maximum * 1.18)

    tight_layout(figure, rect=(0.03, 0, 1.0, 1.0))
    return figure, axes


def generate_direct_token_usage_figure(
    analysis: DirectEfficiencyAnalysis,
    output_dir: Path,
) -> Path:
    """Plot and save average input/output token usage."""
    output_path = output_dir / DIRECT_TOKEN_USAGE_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_direct_token_usage(analysis)

    import matplotlib.pyplot as plt

    save_pgf_figure(figure, output_path)
    plt.close(figure)
    return output_path
