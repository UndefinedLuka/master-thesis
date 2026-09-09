"""Produce the direct-context latency decomposition by serialization format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    DIRECT_LATENCY_FIGURE_FILENAME,
    FORMAT_LABELS,
    LATENCY_PHASE_COLORS,
    LATENCY_PHASE_LABELS,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.direct_efficiency_data import DirectEfficiencyAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_direct_latency(
    analysis: DirectEfficiencyAnalysis,
) -> tuple[Figure, list[Axes]]:
    """Plot mean TTFT plus post-first-token generation, with total-latency labels."""
    summary = analysis.summary
    required = {
        "dataset",
        "format",
        "average_ttft_s",
        "average_generation_time_s",
        "average_total_latency_s",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Latency summary is missing: {', '.join(sorted(missing))}")
    if summary.empty:
        raise ValueError("No direct latency measurements are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    datasets = [value for value in DATASET_ORDER if value in set(summary["dataset"])]
    formats = [
        value for value in SERIALIZATION_FORMAT_ORDER if value in set(summary["format"])
    ]
    figure, axes_array = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(11.0, 4.2 * len(datasets)),
        sharex=True,
        squeeze=False,
    )
    axes = [row[0] for row in axes_array]

    for index, dataset in enumerate(datasets):
        axis = axes[index]
        panel = summary[summary["dataset"].eq(dataset)].set_index("format")
        ttft = [float(panel.loc[value, "average_ttft_s"]) for value in formats]
        generation = [
            float(panel.loc[value, "average_generation_time_s"]) for value in formats
        ]
        totals = [float(panel.loc[value, "average_total_latency_s"]) for value in formats]
        positions = list(range(len(formats)))
        axis.bar(
            positions,
            ttft,
            color=LATENCY_PHASE_COLORS["ttft"],
            width=0.72,
            label=LATENCY_PHASE_LABELS["ttft"],
        )
        axis.bar(
            positions,
            generation,
            bottom=ttft,
            color=LATENCY_PHASE_COLORS["generation"],
            width=0.72,
            label=LATENCY_PHASE_LABELS["generation"],
        )
        maximum = max(totals)
        for position, total in zip(positions, totals, strict=True):
            axis.text(
                position,
                total + maximum * 0.025,
                f"{total:.2f}s",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        axis.set_ylabel("Average latency (seconds)")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.set_ylim(top=maximum * 1.15)
        if index == len(datasets) - 1:
            axis.set_xticks(
                positions,
                [FORMAT_LABELS[value] for value in formats],
                rotation=25,
                ha="right",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Latency phase",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Direct-context response latency by serialization format",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            "Stack height is total response latency: TTFT plus generation after the first "
            "token. Means use the common complete-case cohort."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.03, 0.05, 1.0, 0.90))
    return figure, axes


def generate_direct_latency_figure(
    analysis: DirectEfficiencyAnalysis,
    output_dir: Path,
) -> Path:
    """Plot and save direct response latency by serialization format."""
    output_path = output_dir / DIRECT_LATENCY_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_direct_latency(analysis)

    import matplotlib.pyplot as plt

    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path
