"""Generate per-dataset heatmaps for mean retrieval-accuracy metrics.

Each output figure contains aligned heatmaps for normalized exact match, ROUGE-L F1, and
BERTScore F1. Rows are models and columns are CSV, Markdown, TOON, YAML, JSON, XML, and SQL.
Scores are arithmetic means over all scored records in the selected dataset, including zeroes
assigned to failed outputs. Missing model/method combinations are rendered as hatched ``N/A``
cells rather than zero.

    uv run --extra visualizations python -m visualizations.accuracy_heatmaps
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from config.paths import ProjectPaths
from visualizations.accuracy_table import (
    METRIC_ORDER,
    METRIC_SOURCE_COLUMNS,
    calculate_accuracy_means,
    model_display_name,
    model_sort_key,
)
from visualizations.data import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_TITLES,
    load_scored_results,
)

OUTPUT_STEM = "accuracy_heatmaps"
SCORE_MIN = -0.2
SCORE_MAX = 1.0
HEATMAP_METRICS: tuple[str, ...] = (
    "normalized_exact_match",
    "rouge_l_f",
    "bertscore_f",
)


def calculate_dataset_accuracy_means(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean model/method/metric scores independently for each dataset."""
    required_columns = {
        "dataset",
        "pipeline",
        "format",
        "model_id",
        *METRIC_SOURCE_COLUMNS.values(),
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    pieces: list[pd.DataFrame] = []
    for dataset, dataset_results in results.groupby("dataset", observed=True):
        summary = calculate_accuracy_means(dataset_results)
        if summary.empty:
            continue
        summary = summary[summary["metric"].isin(METRIC_ORDER)].copy()
        summary.insert(0, "dataset", dataset)
        pieces.append(summary)
    if not pieces:
        return _empty_summary()
    return pd.concat(pieces, ignore_index=True)


def plot_accuracy_heatmaps(
    summary: pd.DataFrame,
    *,
    dataset: str,
    methods: Sequence[str] = METHOD_ORDER,
    metrics: Sequence[str] = HEATMAP_METRICS,
) -> tuple[Figure, list[Axes]]:
    """Create one dataset figure containing three annotated accuracy heatmaps."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    required_columns = {"dataset", "model_id", "method", "metric", "mean_score"}
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Accuracy summary is missing required column(s): {names}")

    panel = summary[summary["dataset"].eq(dataset)]
    if panel.empty:
        raise ValueError(f"No accuracy scores are available for dataset: {dataset}")

    selected_methods = tuple(methods)
    selected_metrics = tuple(metrics)
    models = sorted(panel["model_id"].unique(), key=model_sort_key)
    model_labels = [model_display_name(model) for model in models]
    method_labels = [METHOD_LABELS.get(method, method) for method in selected_methods]

    sns.set_theme(style="white", context="talk")
    figure, axes_array = plt.subplots(
        1,
        len(selected_metrics),
        figsize=(max(15.5, len(selected_metrics) * 5.4), max(8.2, len(models) * 0.55)),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_array[0])
    colorbar_axis = figure.add_axes((0.925, 0.17, 0.016, 0.67))
    color_map = sns.color_palette("viridis", as_cmap=True)

    for index, (axis, metric) in enumerate(zip(axes, selected_metrics, strict=True)):
        metric_panel = panel[panel["metric"].eq(metric)]
        matrix = metric_panel.pivot(index="model_id", columns="method", values="mean_score")
        matrix = matrix.reindex(index=models, columns=selected_methods)
        annotations = matrix.map(lambda value: "N/A" if pd.isna(value) else f"{value:.3f}")
        missing_mask = matrix.isna()

        sns.heatmap(
            matrix,
            mask=missing_mask,
            annot=annotations,
            fmt="",
            cmap=color_map,
            vmin=SCORE_MIN,
            vmax=SCORE_MAX,
            linewidths=0.8,
            linecolor="white",
            cbar=index == len(selected_metrics) - 1,
            cbar_ax=colorbar_axis if index == len(selected_metrics) - 1 else None,
            ax=axis,
            annot_kws={"fontsize": 8.5},
        )
        _draw_missing_cells(axis, missing_mask)
        axis.set_title(METRIC_TITLES.get(metric, metric), weight="bold", pad=12)
        axis.set_xlabel("Method")
        axis.set_ylabel("Model" if index == 0 else "")
        axis.set_xticklabels(method_labels, rotation=35, ha="right")
        axis.set_yticklabels(model_labels, rotation=0)
        axis.tick_params(axis="both", labelsize=9, labelleft=index == 0)

    colorbar_axis.set_ylabel("Mean score", rotation=270, labelpad=22)
    colorbar_axis.yaxis.set_major_formatter(lambda value, _position: f"{value:.1f}")
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    figure.suptitle(f"Mean Retrieval Accuracy — {dataset_label}", weight="bold", y=0.99)
    figure.text(
        0.5,
        0.015,
        (
            "Cell values are arithmetic means over all scored questions; failed outputs retain "
            "zero scores. All panels use a common -0.2 to 1.0 scale; hatched cells are N/A."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.16, right=0.90, bottom=0.15, top=0.88, wspace=0.08)
    return figure, axes


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def output_path_for_dataset(output_dir: Path, dataset: str) -> Path:
    """Return the deterministic PNG path for a dataset heatmap figure."""
    return output_dir / f"{OUTPUT_STEM}_{dataset}.png"


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Generate separate BIRD and Spider mean-accuracy heatmap figures"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=project_paths.results_dir,
        help="scored NDJSON file or result directory (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_paths.root / "analysis" / "figures",
        help="directory for dataset PNGs (default: analysis/figures/)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_ORDER,
        default=None,
        help="dataset to include; repeat to include both (default: bird and spider)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="exact model_id to include; repeat to include multiple models (default: all)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the generated plots interactively after saving them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load scored results and generate one three-metric heatmap figure per dataset."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    selected_datasets = _canonical_selection(DATASET_ORDER, args.dataset)
    results = load_scored_results(
        args.results,
        columns=(
            "dataset",
            "pipeline",
            "format",
            "model_id",
            *METRIC_SOURCE_COLUMNS.values(),
        ),
        datasets=selected_datasets,
        pipelines=("direct", "text_to_sql"),
        models=args.model,
    )
    summary = calculate_dataset_accuracy_means(results)
    if summary.empty:
        parser.error("no scored accuracy records matched the selected filters")

    generated: list[tuple[Figure, Path]] = []
    for dataset in selected_datasets:
        if not summary["dataset"].eq(dataset).any():
            print(f"Skipping {dataset}: no scored accuracy records")
            continue
        figure, _ = plot_accuracy_heatmaps(summary, dataset=dataset)
        output_path = output_path_for_dataset(args.output_dir, dataset)
        save_figure(figure, output_path)
        generated.append((figure, output_path))
        print(f"Saved {DATASET_LABELS.get(dataset, dataset)} heatmaps to {output_path}")

    if not generated:
        parser.error("no selected dataset had scored accuracy records")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        for figure, _ in generated:
            plt.close(figure)
    return 0


def _draw_missing_cells(axis: Axes, missing_mask: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    for row_index in range(missing_mask.shape[0]):
        for column_index in range(missing_mask.shape[1]):
            if not missing_mask.iat[row_index, column_index]:
                continue
            axis.add_patch(
                Rectangle(
                    (column_index, row_index),
                    1,
                    1,
                    facecolor="#eeeeee",
                    edgecolor="#999999",
                    hatch="///",
                    linewidth=0.8,
                )
            )
            axis.text(
                column_index + 0.5,
                row_index + 0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=8,
                color="#555555",
                weight="bold",
            )


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "dataset",
            "model_id",
            "method",
            "metric",
            "mean_score",
            "sample_count",
            "datasets",
            "rank",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
