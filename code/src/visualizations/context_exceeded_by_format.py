"""Plot unique questions that exceeded the model context, split by dataset and format.

A question can appear in many model result files. This visualization counts each ``example_id``
once per dataset/format when at least one selected model recorded ``error=context_exceeded``;
it does not count model-run rows. Each format is one stacked bar split into BIRD and Spider.

    uv run --extra visualizations python -m visualizations.context_exceeded_by_format
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
from serialization.types import FORMATS
from visualizations.data import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    SQL_EXECUTION_LABEL,
    load_scored_results,
)

OUTPUT_FILENAME = "context_exceeded_by_format.png"
CONTEXT_EXCEEDED_ERROR = "context_exceeded"
FORMAT_ORDER: tuple[str, ...] = tuple(
    method for method in METHOD_ORDER if method != SQL_EXECUTION_LABEL
)



def calculate_context_exceeded_counts(
    results: pd.DataFrame,
    *,
    datasets: Sequence[str] = DATASET_ORDER,
    formats: Sequence[str] = FORMAT_ORDER,
) -> pd.DataFrame:
    """Count unique context-exceeded questions for every dataset/format combination."""
    required_columns = {"example_id", "dataset", "pipeline", "format", "error"}
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    dataset_order = tuple(dict.fromkeys(datasets))
    format_order = tuple(dict.fromkeys(formats))
    if not dataset_order or not format_order:
        raise ValueError("At least one dataset and format must be selected")

    data = results.loc[:, sorted(required_columns)].copy()
    exceeded = data[
        data["pipeline"].eq("direct")
        & data["error"].eq(CONTEXT_EXCEEDED_ERROR)
        & data["dataset"].isin(dataset_order)
        & data["format"].isin(format_order)
    ].dropna(subset=["example_id"])

    counts = exceeded.groupby(["format", "dataset"], observed=True)["example_id"].nunique()
    combinations = pd.MultiIndex.from_product(
        [format_order, dataset_order], names=["format", "dataset"]
    )
    summary = (
        counts.reindex(combinations, fill_value=0)
        .rename("question_count")
        .reset_index()
    )
    summary["question_count"] = summary["question_count"].astype(int)
    return summary


def plot_context_exceeded_counts(summary: pd.DataFrame) -> tuple[Figure, Axes]:
    """Create a stacked BIRD/Spider bar for each serialization format."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import StrMethodFormatter

    required_columns = {"format", "dataset", "question_count"}
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Context-exceeded summary is missing required column(s): {names}")
    if summary.empty:
        raise ValueError("No dataset/format combinations are available to plot")

    format_order = list(dict.fromkeys(summary["format"].tolist()))
    dataset_order = list(dict.fromkeys(summary["dataset"].tolist()))
    pivot = (
        summary.pivot(index="format", columns="dataset", values="question_count")
        .reindex(index=format_order, columns=dataset_order, fill_value=0)
        .fillna(0)
    )

    sns.set_theme(style="whitegrid", context="talk")
    figure_width = max(9.5, len(format_order) * 1.45)
    figure, axes = plt.subplots(figsize=(figure_width, 6.8))
    colors = sns.color_palette("colorblind", n_colors=len(dataset_order))
    x_positions = list(range(len(format_order)))
    bottoms = pd.Series(0.0, index=format_order)
    totals = pivot.sum(axis=1)
    maximum = max(float(totals.max()), 1.0)

    for dataset, color in zip(dataset_order, colors, strict=True):
        values = pivot[dataset].astype(float)
        axes.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=0.76,
            label=DATASET_LABELS.get(dataset, dataset),
            color=color,
            edgecolor="white",
            linewidth=0.9,
        )
        for position, bottom, value in zip(x_positions, bottoms, values, strict=True):
            if value > 0:
                axes.text(
                    position,
                    bottom + value / 2,
                    f"{int(value):,}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    weight="bold",
                )
            else:
                axes.text(
                    position,
                    bottom + maximum * 0.015,
                    "0",
                    ha="center",
                    va="bottom",
                    color=color,
                    fontsize=9,
                    weight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 0.8,
                    },
                )
        bottoms = bottoms + values

    axes.set_title("Unique Questions Exceeding Context by Method", pad=18, weight="bold")
    axes.set_xlabel("Method")
    axes.set_ylabel("Questions exceeding context")
    axes.set_xticks(
        x_positions,
        [METHOD_LABELS.get(method, method) for method in format_order],
    )
    axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes.grid(axis="x", visible=False)
    axes.legend(title="Dataset", frameon=False, ncols=len(dataset_order))
    axes.set_ylim(top=maximum * 1.1)
    sns.despine(ax=axes)

    figure.text(
        0.5,
        0.01,
        (
            "Each benchmark question is counted once per method, even when multiple models "
            "exceeded the context window."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure, axes


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Plot unique context-exceeded questions by format, split by BIRD and Spider"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=project_paths.results_dir,
        help="scored NDJSON file or result directory (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_paths.root / "analysis" / "figures" / OUTPUT_FILENAME,
        help=f"output image path (default: analysis/figures/{OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_ORDER,
        default=None,
        help="dataset to include; repeat to include both (default: bird and spider)",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=FORMATS,
        default=None,
        help="format to include; repeat to include multiple formats (default: all)",
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
        help="open the plot interactively after saving it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load results, count unique context overflows, and save the visualization."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    selected_datasets = _canonical_selection(DATASET_ORDER, args.dataset)
    selected_formats = _canonical_selection(FORMAT_ORDER, args.formats)
    results = load_scored_results(
        args.results,
        columns=("example_id", "dataset", "pipeline", "format", "error"),
        datasets=selected_datasets,
        pipelines=("direct",),
        formats=selected_formats,
        models=args.model,
    )
    if results.empty:
        parser.error("no direct-pipeline result records matched the selected filters")

    summary = calculate_context_exceeded_counts(
        results,
        datasets=selected_datasets,
        formats=selected_formats,
    )
    figure, _ = plot_context_exceeded_counts(summary)
    save_figure(figure, args.output)

    printable = summary.pivot(
        index="format", columns="dataset", values="question_count"
    ).reindex(index=selected_formats, columns=selected_datasets)
    printable = printable.rename(columns=DATASET_LABELS)
    print(printable.to_string())
    print(f"Saved figure to {args.output}")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


if __name__ == "__main__":
    raise SystemExit(main())
