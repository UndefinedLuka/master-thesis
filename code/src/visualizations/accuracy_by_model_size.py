"""Generate per-dataset accuracy-scaling plots by model family and size.

Each dataset figure has three metric rows (normalized exact match, ROUGE-L F1, and BERTScore
F1) and three family columns (Gemma, Qwen, and Ministral). The x-axis is the nominal parameter
count encoded by the model identifier, on a logarithmic scale; each line is one serialization
method or SQL. Missing experiment combinations remain gaps rather than being treated as zero.

Accuracy means include every scored question, including zeroes assigned to failed outputs.
Gemma ``26B-A4B`` is positioned at 26B total parameters and labeled with its 4B active size.

    uv run --extra visualizations python -m visualizations.accuracy_by_model_size
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from config.paths import ProjectPaths
from visualizations.accuracy_heatmaps import (
    HEATMAP_METRICS,
    calculate_dataset_accuracy_means,
)
from visualizations.accuracy_table import METRIC_SOURCE_COLUMNS
from visualizations.data import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_TITLES,
    load_scored_results,
)

OUTPUT_STEM = "accuracy_by_model_size"
FAMILY_ORDER: tuple[str, ...] = ("Gemma", "Qwen", "Ministral")
MODEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "Gemma": re.compile(r"^gemma-4-(?:E)?(?P<size>\d+(?:\.\d+)?)B", re.IGNORECASE),
    "Qwen": re.compile(r"^Qwen3\.5-(?P<size>\d+(?:\.\d+)?)B", re.IGNORECASE),
    "Ministral": re.compile(
        r"^Ministral-3-(?P<size>\d+(?:\.\d+)?)B", re.IGNORECASE
    ),
}


def calculate_model_size_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-dataset means and attach parsed model family/size metadata."""
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

    summary = calculate_dataset_accuracy_means(results)
    if summary.empty:
        return _empty_summary()

    metadata = summary["model_id"].map(parse_model_family_size)
    summary = summary[metadata.notna()].copy()
    if summary.empty:
        return _empty_summary()
    summary["model_family"] = [value[0] for value in metadata[metadata.notna()]]
    summary["model_size_b"] = [value[1] for value in metadata[metadata.notna()]]
    summary["size_label"] = summary["model_id"].map(model_size_label)
    return summary.sort_values(
        ["dataset", "model_family", "model_size_b", "method", "metric"]
    ).reset_index(drop=True)


def parse_model_family_size(model_id: str) -> tuple[str, float] | None:
    """Parse a supported model family and nominal parameter count in billions."""
    for family in FAMILY_ORDER:
        match = MODEL_PATTERNS[family].search(model_id)
        if match:
            return family, float(match.group("size"))
    return None


def model_size_label(model_id: str) -> str:
    """Return a concise tick label, including active parameters for mixture models."""
    metadata = parse_model_family_size(model_id)
    if metadata is None:
        return model_id
    _, size = metadata
    label = _format_size(size)
    active_match = re.search(r"-A(?P<active>\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
    if active_match:
        label += f"\n(A{active_match.group('active')} active)"
    return label


def plot_accuracy_by_model_size(
    summary: pd.DataFrame,
    *,
    dataset: str,
    families: Sequence[str] = FAMILY_ORDER,
    metrics: Sequence[str] = HEATMAP_METRICS,
    methods: Sequence[str] = METHOD_ORDER,
) -> tuple[Figure, list[list[Axes]]]:
    """Create one metric-by-family scaling figure for a dataset."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    required_columns = {
        "dataset",
        "model_id",
        "model_family",
        "model_size_b",
        "size_label",
        "method",
        "metric",
        "mean_score",
    }
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Model-size accuracy summary is missing required column(s): {names}")

    panel = summary[summary["dataset"].eq(dataset)].copy()
    if panel.empty:
        raise ValueError(f"No model-size accuracy scores are available for dataset: {dataset}")

    selected_families = [family for family in families if family in set(panel["model_family"])]
    selected_metrics = tuple(metrics)
    selected_methods = [method for method in methods if method in set(panel["method"])]
    if not selected_families or not selected_metrics or not selected_methods:
        raise ValueError("No recognized families, metrics, or methods are available to plot")

    sns.set_theme(style="whitegrid", context="talk")
    figure, axes_array = plt.subplots(
        len(selected_metrics),
        len(selected_families),
        figsize=(17.5, 12.0),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    axes = [list(row) for row in axes_array]
    palette = dict(
        zip(
            selected_methods,
            sns.color_palette("colorblind", n_colors=len(selected_methods)),
            strict=True,
        )
    )

    for column_index, family in enumerate(selected_families):
        family_panel = panel[panel["model_family"].eq(family)]
        model_metadata = (
            family_panel[["model_id", "model_size_b", "size_label"]]
            .drop_duplicates()
            .sort_values(["model_size_b", "model_id"])
        )
        model_ids = model_metadata["model_id"].tolist()
        x_values = model_metadata["model_size_b"].tolist()
        tick_labels = model_metadata["size_label"].tolist()

        for row_index, metric in enumerate(selected_metrics):
            axis = axes[row_index][column_index]
            metric_panel = family_panel[family_panel["metric"].eq(metric)]
            for method in selected_methods:
                method_scores = (
                    metric_panel[metric_panel["method"].eq(method)]
                    .set_index("model_id")["mean_score"]
                )
                y_values = [method_scores.get(model_id, math.nan) for model_id in model_ids]
                if all(pd.isna(value) for value in y_values):
                    continue
                axis.plot(
                    x_values,
                    y_values,
                    color=palette[method],
                    marker="o",
                    markersize=5.0,
                    linewidth=2.0,
                    alpha=0.9,
                )

            axis.set_xscale("log")
            axis.set_xticks(x_values, tick_labels)
            axis.minorticks_off()
            axis.grid(axis="x", alpha=0.3)
            axis.grid(axis="y", alpha=0.55)
            axis.yaxis.set_major_locator(MultipleLocator(0.2))
            axis.set_ylim(-0.2 if metric == "bertscore_f" else 0.0, 1.0)
            if row_index == 0:
                axis.set_title(family, weight="bold", pad=12)
            if column_index == 0:
                axis.set_ylabel(METRIC_TITLES.get(metric, metric))
            else:
                axis.set_ylabel("")
            if row_index == len(selected_metrics) - 1:
                axis.set_xlabel("Nominal model size (B parameters, log scale)")
                axis.tick_params(axis="x", labelsize=8.5)
            else:
                axis.set_xlabel("")
            sns.despine(ax=axis)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=palette[method],
            marker="o",
            linewidth=2,
            markersize=6,
            label=METHOD_LABELS.get(method, method),
        )
        for method in selected_methods
    ]
    figure.legend(
        handles=method_handles,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncols=len(method_handles),
        frameon=False,
        fontsize=9,
    )
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    figure.suptitle(f"Accuracy by Model Size and Family — {dataset_label}", weight="bold", y=0.995)
    figure.text(
        0.5,
        0.012,
        (
            "Points are mean scores over all questions, including zero-scored failures. "
            "Missing experiment combinations appear as gaps; Gemma 26B-A4B is positioned at "
            "26B total parameters (4B active)."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.84, hspace=0.16, wspace=0.10)
    return figure, axes


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def output_path_for_dataset(output_dir: Path, dataset: str) -> Path:
    """Return the deterministic PNG path for a dataset scaling figure."""
    return output_dir / f"{OUTPUT_STEM}_{dataset}.png"


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Generate separate BIRD and Spider accuracy-by-model-size figures"
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
    """Load scored results and generate one family-scaling figure per dataset."""
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
    summary = calculate_model_size_accuracy(results)
    if summary.empty:
        parser.error("no supported model-size accuracy records matched the selected filters")

    generated: list[tuple[Figure, Path]] = []
    for dataset in selected_datasets:
        if not summary["dataset"].eq(dataset).any():
            print(f"Skipping {dataset}: no supported model-family results")
            continue
        figure, _ = plot_accuracy_by_model_size(summary, dataset=dataset)
        output_path = output_path_for_dataset(args.output_dir, dataset)
        save_figure(figure, output_path)
        generated.append((figure, output_path))
        print(f"Saved {DATASET_LABELS.get(dataset, dataset)} scaling figure to {output_path}")

    if not generated:
        parser.error("no selected dataset had supported model-family results")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        for figure, _ in generated:
            plt.close(figure)
    return 0


def _format_size(size: float) -> str:
    return str(int(size)) if size.is_integer() else f"{size:g}"


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
            "model_family",
            "model_size_b",
            "size_label",
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
