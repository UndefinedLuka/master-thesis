"""Produce model-level Text-to-SQL input/output token figures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    TEXT_TO_SQL_TOKEN_FIGURE_FILENAME,
    TOKENIZER_FAMILY_COLORS,
)
from visualizations.plot_numbers import number_formatter, save_pgf_figure, tight_layout
from visualizations.text_to_sql_data import TextToSqlAnalysis, text_to_sql_model_label
from visualizations.uncertainty import interval_marks

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

TOKEN_METRICS: tuple[tuple[str, str, str], ...] = (
    ("average_input_tokens", "Average input tokens", ",.0f"),
    ("average_output_tokens", "Average output tokens", ",.0f"),
)


def plot_text_to_sql_token_usage(
    analysis: TextToSqlAnalysis,
) -> tuple[Figure, list[list[Axes]]]:
    """Plot average accumulated input/output tokens for every model and dataset."""
    summary = analysis.model_summary
    if summary.empty:
        raise ValueError("No Text-to-SQL token values are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes_array = plt.subplots(
        nrows=len(DATASET_ORDER),
        ncols=len(TOKEN_METRICS),
        figsize=(15.2, 8.5),
        squeeze=False,
    )
    axes = [list(row) for row in axes_array]
    for row_index, dataset in enumerate(DATASET_ORDER):
        rows = _ordered_models(summary[summary["dataset"].eq(dataset)])
        if rows.empty:
            for axis in axes[row_index]:
                axis.set_visible(False)
            continue
        labels = [
            text_to_sql_model_label(row.model_family, row.size_label)
            for row in rows.itertuples()
        ]
        colors = [TOKENIZER_FAMILY_COLORS[value] for value in rows["model_family"]]
        for column_index, (metric, title, _number_format) in enumerate(TOKEN_METRICS):
            axis = axes[row_index][column_index]
            values = rows[metric].astype(float).tolist()
            axis.bar(range(len(rows)), values, color=colors, width=0.75)
            interval_marks(axis, range(len(rows)), rows, metric)
            axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(title, fontweight="bold")
            axis.set_ylabel(DATASET_LABELS[dataset] if column_index == 0 else "")
            axis.yaxis.set_major_formatter(number_formatter())
            axis.set_xticks(range(len(rows)), labels, rotation=65, ha="right")
            maximum = max(values)
            if f"{metric}_high" in rows:
                maximum = max(maximum, float(rows[f"{metric}_high"].max()))
            axis.set_ylim(top=maximum * 1.15)

    tight_layout(figure, rect=(0.03, 0, 1.0, 1.0))
    return figure, axes


def generate_text_to_sql_token_figure(
    analysis: TextToSqlAnalysis,
    output_dir: Path,
) -> Path:
    """Plot and save Text-to-SQL token usage."""
    output_path = output_dir / TEXT_TO_SQL_TOKEN_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_text_to_sql_token_usage(analysis)

    import matplotlib.pyplot as plt

    save_pgf_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _ordered_models(summary):
    positions = {value: index for index, value in enumerate(TOKENIZER_FAMILY_COLORS)}
    return summary.sort_values(
        ["model_family", "model_size_b"],
        key=lambda values: values.map(positions) if values.name == "model_family" else values,
    ).reset_index(drop=True)
