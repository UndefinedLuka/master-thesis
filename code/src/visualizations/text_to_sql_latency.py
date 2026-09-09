"""Produce retry-aware Text-to-SQL latency decomposition by model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    TEXT_TO_SQL_LATENCY_FIGURE_FILENAME,
    TEXT_TO_SQL_LATENCY_PHASE_COLORS,
    TEXT_TO_SQL_LATENCY_PHASE_LABELS,
    TOKENIZER_FAMILY_COLORS,
)
from visualizations.text_to_sql_data import TextToSqlAnalysis, text_to_sql_model_label

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PHASES: tuple[tuple[str, str], ...] = (
    ("average_cumulative_ttft_s", "ttft"),
    ("average_generation_time_s", "generation"),
    ("average_sql_execution_time_s", "sql"),
)


def plot_text_to_sql_latency(
    analysis: TextToSqlAnalysis,
) -> tuple[Figure, list[Axes]]:
    """Plot accumulated latency phases for each model, annotating mean retry counts."""
    summary = analysis.model_summary
    if summary.empty:
        raise ValueError("No Text-to-SQL latency values are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes_array = plt.subplots(
        nrows=len(DATASET_ORDER),
        ncols=1,
        figsize=(13.5, 8.5),
        squeeze=False,
    )
    axes = [row[0] for row in axes_array]
    for index, dataset in enumerate(DATASET_ORDER):
        axis = axes[index]
        rows = _ordered_models(summary[summary["dataset"].eq(dataset)])
        if rows.empty:
            axis.set_visible(False)
            continue
        positions = list(range(len(rows)))
        bottoms = [0.0] * len(rows)
        for metric, phase in PHASES:
            values = rows[metric].astype(float).tolist()
            axis.bar(
                positions,
                values,
                bottom=bottoms,
                width=0.75,
                color=TEXT_TO_SQL_LATENCY_PHASE_COLORS[phase],
                label=TEXT_TO_SQL_LATENCY_PHASE_LABELS[phase],
            )
            bottoms = [left + value for left, value in zip(bottoms, values, strict=True)]
        maximum = max(rows["average_total_latency_s"])
        for position, total, retries in zip(
            positions,
            rows["average_total_latency_s"],
            rows["average_retries"],
            strict=True,
        ):
            axis.text(
                position,
                float(total) + maximum * 0.025,
                f"{float(total):.2f}s\n{float(retries):.2f} retries",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        labels = [
            text_to_sql_model_label(row.model_family, row.size_label)
            for row in rows.itertuples()
        ]
        axis.set_xticks(positions, labels, rotation=40, ha="right", fontsize=7.5)
        axis.set_ylabel(f"{DATASET_LABELS[dataset]}\nAverage seconds")
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylim(top=maximum * 1.28)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Latency phase",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Text-to-SQL response latency and correction retries",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            "Stacks include all generation attempts and SQL executions. Labels report total "
            "latency and mean retries beyond the first attempt."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.03, 0.05, 1.0, 0.89))
    return figure, axes


def generate_text_to_sql_latency_figure(
    analysis: TextToSqlAnalysis,
    output_dir: Path,
) -> Path:
    """Plot and save retry-aware Text-to-SQL latency."""
    output_path = output_dir / TEXT_TO_SQL_LATENCY_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_text_to_sql_latency(analysis)

    import matplotlib.pyplot as plt

    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _ordered_models(summary):
    positions = {value: index for index, value in enumerate(TOKENIZER_FAMILY_COLORS)}
    return summary.sort_values(
        ["model_family", "model_size_b"],
        key=lambda values: values.map(positions) if values.name == "model_family" else values,
    ).reset_index(drop=True)
