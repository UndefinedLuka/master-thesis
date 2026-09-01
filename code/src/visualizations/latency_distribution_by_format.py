"""Plot the distribution of end-to-end latency by method, faceted by dataset.

Completed answers are shown as box plots with a deterministic sample of individual observations.
The logarithmic time axis keeps sub-second Spider/SQL results and multi-minute BIRD outliers
visible in the same figure.

    uv run --extra visualizations python -m visualizations.latency_distribution_by_format
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

OUTPUT_FILENAME = "latency_distribution_by_format.png"
POINT_SAMPLE_PER_GROUP = 350


def prepare_latency_data(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize method labels and retain valid, positive end-to-end latency observations."""
    required_columns = {"dataset", "pipeline", "format", "total_time_s"}
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    data = results.loc[:, sorted(required_columns)].copy()
    data["method"] = data["format"]
    is_sql = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql, "method"] = SQL_EXECUTION_LABEL
    data["total_time_s"] = pd.to_numeric(data["total_time_s"], errors="coerce")
    data = data.dropna(subset=["dataset", "method", "total_time_s"])
    data = data[data["total_time_s"] > 0].copy()
    data["method"] = pd.Categorical(data["method"], categories=METHOD_ORDER, ordered=True)
    return data.sort_values(["dataset", "method"]).reset_index(drop=True)


def plot_latency_distribution(data: pd.DataFrame) -> tuple[Figure, list[Axes]]:
    """Create BIRD/Spider latency-distribution panels with a shared logarithmic time axis."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter, LogLocator

    required_columns = {"dataset", "method", "total_time_s"}
    missing = required_columns.difference(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Latency data is missing required column(s): {names}")
    if data.empty:
        raise ValueError("No completed positive latency observations are available to plot")

    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(data["dataset"])]
    methods = [method for method in METHOD_ORDER if method in set(data["method"].dropna())]
    if not datasets or not methods:
        raise ValueError("No recognized datasets or methods are available to plot")

    sns.set_theme(style="whitegrid", context="talk")
    figure, axes_array = plt.subplots(
        1,
        len(datasets),
        figsize=(max(12.5, len(datasets) * 7.0), 7.0),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_array[0])
    palette = dict(
        zip(methods, sns.color_palette("colorblind", n_colors=len(methods)), strict=True)
    )

    for axes_index, (axis, dataset) in enumerate(zip(axes, datasets, strict=True)):
        panel = data[data["dataset"].eq(dataset)]
        panel_methods = [method for method in methods if method in set(panel["method"].dropna())]
        sns.boxplot(
            data=panel,
            x="method",
            y="total_time_s",
            order=panel_methods,
            hue="method",
            hue_order=panel_methods,
            palette=palette,
            showfliers=False,
            width=0.65,
            linewidth=1.0,
            legend=False,
            ax=axis,
        )
        sampled = _sample_points(panel)
        sns.stripplot(
            data=sampled,
            x="method",
            y="total_time_s",
            order=panel_methods,
            hue="method",
            hue_order=panel_methods,
            palette=palette,
            jitter=0.24,
            size=2.2,
            alpha=0.22,
            linewidth=0,
            legend=False,
            ax=axis,
        )
        axis.set_yscale("log")
        axis.set_title(DATASET_LABELS.get(dataset, dataset), weight="bold")
        axis.set_xlabel("Method")
        axis.set_ylabel("End-to-end latency (seconds)" if axes_index == 0 else "")
        axis.set_xticks(
            range(len(panel_methods)),
            [METHOD_LABELS.get(method, method) for method in panel_methods],
            rotation=25,
            ha="right",
        )
        axis.yaxis.set_major_locator(LogLocator(base=10))
        axis.yaxis.set_major_formatter(FuncFormatter(_format_log_seconds))
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)

    figure.suptitle("Latency Distribution by Method and Dataset", weight="bold", y=0.99)
    figure.text(
        0.5,
        0.015,
        (
            "Boxes show the median and interquartile range; points are a deterministic sample. "
            "The time axis is logarithmic."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.96))
    return figure, axes


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Plot completed-answer latency distributions by method and dataset"
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
        help="direct format to include; repeat for multiple formats (default: all)",
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
    """Load completed results and save the latency-distribution visualization."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    selected_datasets = _canonical_selection(DATASET_ORDER, args.dataset)
    results = load_scored_results(
        args.results,
        columns=("dataset", "pipeline", "format", "total_time_s"),
        datasets=selected_datasets,
        pipelines=("direct", "text_to_sql"),
        models=args.model,
        answers_only=True,
    )
    if args.formats is not None:
        keep_direct = results["format"].isin(args.formats)
        keep_sql = results["pipeline"].eq("text_to_sql")
        results = results[keep_direct | keep_sql]

    data = prepare_latency_data(results)
    if data.empty:
        parser.error("no completed positive latency records matched the selected filters")

    figure, _ = plot_latency_distribution(data)
    save_figure(figure, args.output)
    print(_latency_summary(data).to_string(index=False))
    print(f"Saved figure to {args.output}")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


def _sample_points(data: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        group.sample(n=min(len(group), POINT_SAMPLE_PER_GROUP), random_state=42)
        for _, group in data.groupby(["dataset", "method"], observed=True)
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else data.iloc[0:0].copy()


def _latency_summary(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby(["dataset", "method"], observed=True)["total_time_s"].agg(
        count="size",
        median_s="median",
        p90_s=lambda values: values.quantile(0.90),
        p95_s=lambda values: values.quantile(0.95),
    ).reset_index()


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


def _format_log_seconds(value: float, _position: float) -> str:
    if value >= 1:
        return f"{value:,.0f}s"
    return f"{value:g}s"


if __name__ == "__main__":
    raise SystemExit(main())
