"""Produce a per-model heatmap of matched direct/SQL mean latency ratios."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualizations.common_cohort_data import CommonCohortAnalysis
from visualizations.common_cohort_plotting import cohort_datasets, panel_title, save_figure
from visualizations.config import (
    COMMON_COHORT_MODEL_FIGURE_FILENAME,
    FORMAT_LABELS,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.direct_accuracy_data import model_size_label, parse_model_family_size
from visualizations.plot_numbers import number
from visualizations.text_to_sql_data import text_to_sql_model_label

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_common_cohort_latency_by_model(analysis: CommonCohortAnalysis) -> Figure:
    """Keep model and question composition identical in each direct/SQL ratio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    datasets = cohort_datasets(analysis)
    direct = analysis.model_summary[
        analysis.model_summary["method"].isin(SERIALIZATION_FORMAT_ORDER)
    ]
    models = sorted(
        direct["model_id"].unique(),
        key=lambda model: (
            TOKENIZER_FAMILY_ORDER.index(parse_model_family_size(model)[0]),
            parse_model_family_size(model)[1],
            model,
        ),
    )
    formats = [value for value in SERIALIZATION_FORMAT_ORDER if value in set(direct["method"])]
    log_ratios = np.log2(direct["latency_ratio"].to_numpy())
    limit = max(1.0, float(np.abs(log_ratios).max()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(12.2, max(4.8, 0.48 * len(models) + 1.5)),
        squeeze=False,
    )
    for axis, dataset in zip(axes[0], datasets, strict=True):
        panel = (
            direct[direct["dataset"].eq(dataset)]
            .pivot(index="model_id", columns="method", values="latency_ratio")
            .reindex(index=models, columns=formats)
        )
        values = panel.to_numpy()
        image = axis.imshow(np.log2(values), cmap="RdBu_r", norm=norm, aspect="auto")
        for row in range(len(models)):
            for col in range(len(formats)):
                value = values[row, col]
                if not np.isfinite(value):
                    continue
                color = "white" if abs(np.log2(value)) > limit * 0.55 else "#18212B"
                axis.text(
                    col, row, number(value, 1) + r"\ensuremath{\times}",
                    ha="center", va="center", fontsize=8.5, color=color
                )
        labels = [
            text_to_sql_model_label(
                parse_model_family_size(model)[0], model_size_label(model)
            ).replace("\n", " ")
            for model in models
        ]
        # Preserve the E2B/E4B variant identifiers rather than labeling them as dense models.
        labels = [
            label.replace("Gemma 4 ", "Gemma 4 E", 1)
            if model.lower().startswith("gemma-4-e")
            else label
            for model, label in zip(models, labels, strict=True)
        ]
        axis.set_yticks(range(len(models)), labels, fontsize=8.5)
        axis.set_xticks(
            range(len(formats)),
            [FORMAT_LABELS[f] for f in formats],
            rotation=30,
            ha="right",
            fontsize=9,
        )
        axis.set_title(panel_title(analysis, dataset), fontsize=11, fontweight="bold", pad=14)
        axis.set_xticks(np.arange(len(formats) + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.subplots_adjust(left=0.16, right=0.88, bottom=0.18, top=0.94, wspace=0.65)
    color_axis = figure.add_axes((0.91, 0.23, 0.016, 0.55))
    colorbar = figure.colorbar(image, cax=color_axis)
    ticks = [float(value) for value in (-4, -2, 0, 2, 4, 6, 8) if -limit <= value <= limit]
    colorbar.set_ticks(
        ticks,
        labels=[number(2**value) + r"\ensuremath{\times}" for value in ticks],
    )
    colorbar.set_label("Direct / SQL mean latency (log colour scale)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    return figure


def generate_common_cohort_model_figure(
    analysis: CommonCohortAnalysis,
    output_dir: Path,
) -> Path:
    return save_figure(
        plot_common_cohort_latency_by_model(analysis),
        output_dir,
        COMMON_COHORT_MODEL_FIGURE_FILENAME,
    )
