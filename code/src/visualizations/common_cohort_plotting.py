"""Shared publication styling for the matched cross-pipeline figures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualizations.common_cohort_data import CommonCohortAnalysis
from visualizations.config import COMMON_COHORT_METHOD_ORDER, DATASET_LABELS, DATASET_ORDER

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def cohort_datasets(analysis: CommonCohortAnalysis) -> list[str]:
    return [value for value in DATASET_ORDER if value in analysis.question_counts]


def cohort_methods(analysis: CommonCohortAnalysis, dataset: str) -> list[str]:
    present = set(analysis.summary.loc[analysis.summary["dataset"].eq(dataset), "method"])
    return [value for value in COMMON_COHORT_METHOD_ORDER if value in present]


def panel_title(analysis: CommonCohortAnalysis, dataset: str) -> str:
    return (
        f"{DATASET_LABELS[dataset]}"
        f"  ·  {analysis.question_counts[dataset]:,} questions"
        f"  ·  {analysis.model_count} models"
    )


def style_axis(axis: Axes, *, grid_axis: str = "x") -> None:
    axis.grid(axis=grid_axis, color="#DADDE1", linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis="both", labelsize=9, length=0)


def save_figure(figure: Figure, output_dir: Path, filename: str) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path
