"""Generate per-dataset accuracy-efficiency Pareto plots.

Each dataset figure contains six panels: mean input-token cost and mean total latency against
normalized exact match, ROUGE-L F1, and BERTScore F1. Every point is one model/method
combination. Color identifies the method, marker shape identifies the model family, and the
non-dominated Pareto frontier is outlined, connected, and annotated.

Accuracy means include all scored questions (including zeroes assigned to failed outputs).
Efficiency means include records with positive token and time measurements, avoiding artificial
zero cost for requests rejected before model invocation.

    uv run --extra visualizations python -m visualizations.accuracy_efficiency_pareto
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
    METRIC_SOURCE_COLUMNS,
    model_display_name,
    model_sort_key,
)
from visualizations.data import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_TITLES,
    SQL_EXECUTION_LABEL,
    load_scored_results,
)

OUTPUT_STEM = "accuracy_efficiency_pareto"
PARETO_METRICS: tuple[str, ...] = (
    "normalized_exact_match",
    "rouge_l_f",
    "bertscore_f",
)
EFFICIENCY_AXES: tuple[tuple[str, str], ...] = (
    ("mean_input_tokens", "Mean input tokens"),
    ("mean_total_time_s", "Mean total latency (seconds)"),
)
FAMILY_ORDER: tuple[str, ...] = ("Ministral", "Qwen", "Gemma", "Other")
FAMILY_MARKERS: dict[str, str] = {
    "Ministral": "o",
    "Qwen": "s",
    "Gemma": "^",
    "Other": "D",
}


def calculate_accuracy_efficiency_points(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one accuracy/efficiency point per dataset, model, and method."""
    required_columns = {
        "dataset",
        "pipeline",
        "format",
        "model_id",
        "input_tokens",
        "total_time_s",
        *METRIC_SOURCE_COLUMNS.values(),
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    data = results.loc[:, sorted(required_columns)].copy()
    data["method"] = data["format"]
    is_sql = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql, "method"] = SQL_EXECUTION_LABEL
    data = data.dropna(subset=["dataset", "model_id", "method"])

    renamed = data.rename(
        columns={source: metric for metric, source in METRIC_SOURCE_COLUMNS.items()}
    )
    for column in (*PARETO_METRICS, "input_tokens", "total_time_s"):
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")

    group_columns = ["dataset", "model_id", "method"]
    accuracy = renamed.groupby(group_columns, as_index=False, observed=True).agg(
        normalized_exact_match=("normalized_exact_match", "mean"),
        rouge_l_f=("rouge_l_f", "mean"),
        bertscore_f=("bertscore_f", "mean"),
        accuracy_sample_count=("normalized_exact_match", "count"),
    )

    measured = renamed[(renamed["input_tokens"] > 0) & (renamed["total_time_s"] > 0)]
    efficiency = measured.groupby(group_columns, as_index=False, observed=True).agg(
        mean_input_tokens=("input_tokens", "mean"),
        mean_total_time_s=("total_time_s", "mean"),
        efficiency_sample_count=("input_tokens", "size"),
    )

    points = accuracy.merge(efficiency, on=group_columns, how="inner", validate="one_to_one")
    points = points.dropna(subset=[*PARETO_METRICS, "mean_input_tokens", "mean_total_time_s"])
    points = points[
        (points["mean_input_tokens"] > 0) & (points["mean_total_time_s"] > 0)
    ].copy()
    points["model_family"] = points["model_id"].map(model_family)
    return points.sort_values(
        ["dataset", "model_id", "method"],
        key=lambda values: (
            values.map(model_sort_key) if values.name == "model_id" else values
        ),
    ).reset_index(drop=True)


def pareto_frontier(
    points: pd.DataFrame, *, cost_column: str, score_column: str
) -> pd.DataFrame:
    """Return points not dominated by another point with lower cost and higher score."""
    required_columns = {cost_column, score_column}
    missing = required_columns.difference(points.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Pareto data is missing required column(s): {names}")

    valid = points.dropna(subset=[cost_column, score_column]).copy()
    if valid.empty:
        return valid

    frontier_indices: list[int] = []
    for point_index, point in valid.iterrows():
        dominated = (
            (valid[cost_column] <= point[cost_column])
            & (valid[score_column] >= point[score_column])
            & (
                (valid[cost_column] < point[cost_column])
                | (valid[score_column] > point[score_column])
            )
        ).any()
        if not dominated:
            frontier_indices.append(point_index)
    return valid.loc[frontier_indices].sort_values(cost_column).reset_index(drop=True)


def plot_accuracy_efficiency_pareto(
    points: pd.DataFrame,
    *,
    dataset: str,
    metrics: Sequence[str] = PARETO_METRICS,
) -> tuple[Figure, list[list[Axes]]]:
    """Create one six-panel accuracy-efficiency Pareto figure for a dataset."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator

    required_columns = {
        "dataset",
        "model_id",
        "model_family",
        "method",
        "mean_input_tokens",
        "mean_total_time_s",
        *metrics,
    }
    missing = required_columns.difference(points.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Pareto points are missing required column(s): {names}")

    panel = points[points["dataset"].eq(dataset)].copy()
    if panel.empty:
        raise ValueError(f"No accuracy-efficiency points are available for dataset: {dataset}")

    selected_metrics = tuple(metrics)
    methods = [method for method in METHOD_ORDER if method in set(panel["method"])]
    families = [family for family in FAMILY_ORDER if family in set(panel["model_family"])]
    if not methods or not families:
        raise ValueError("No recognized methods or model families are available to plot")

    sns.set_theme(style="whitegrid", context="talk")
    figure, axes_array = plt.subplots(
        len(EFFICIENCY_AXES),
        len(selected_metrics),
        figsize=(18.0, 11.5),
        squeeze=False,
    )
    axes = [list(row) for row in axes_array]
    palette = dict(
        zip(methods, sns.color_palette("colorblind", n_colors=len(methods)), strict=True)
    )

    for row_index, (cost_column, cost_label) in enumerate(EFFICIENCY_AXES):
        for column_index, metric in enumerate(selected_metrics):
            axis = axes[row_index][column_index]
            for family in families:
                family_points = panel[panel["model_family"].eq(family)]
                axis.scatter(
                    family_points[cost_column],
                    family_points[metric],
                    c=[palette[method] for method in family_points["method"]],
                    marker=FAMILY_MARKERS[family],
                    s=62,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3,
                )

            frontier = pareto_frontier(
                panel,
                cost_column=cost_column,
                score_column=metric,
            )
            axis.plot(
                frontier[cost_column],
                frontier[metric],
                color="#222222",
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
                zorder=4,
            )
            _highlight_frontier(
                axis,
                frontier,
                cost_column=cost_column,
                score_column=metric,
                palette=palette,
            )

            axis.set_xscale("log")
            axis.xaxis.set_major_locator(LogLocator(base=10))
            tick_formatter = (
                _format_tokens if cost_column == "mean_input_tokens" else _format_time
            )
            axis.xaxis.set_major_formatter(FuncFormatter(tick_formatter))
            axis.set_xlabel(cost_label)
            axis.set_ylabel("Mean score" if column_index == 0 else "")
            axis.set_ylim(-0.2 if metric == "bertscore_f" else 0.0, 1.0)
            axis.yaxis.set_major_locator(MultipleLocator(0.2))
            axis.grid(which="major", alpha=0.55)
            axis.grid(which="minor", axis="x", alpha=0.16)
            if row_index == 0:
                axis.set_title(METRIC_TITLES.get(metric, metric), weight="bold", pad=12)
            sns.despine(ax=axis)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[method],
            markeredgecolor="white",
            markersize=8,
            label=METHOD_LABELS.get(method, method),
        )
        for method in methods
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=8,
            label=family,
        )
        for family in families
    ]
    frontier_handle = Line2D(
        [0],
        [0],
        color="#222222",
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#222222",
        label="Pareto frontier",
    )
    method_legend = figure.legend(
        handles=method_handles,
        title="Method (color)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=len(method_handles),
        frameon=False,
        fontsize=9,
    )
    figure.add_artist(method_legend)
    figure.legend(
        handles=[*family_handles, frontier_handle],
        title="Model family (marker)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncols=len(family_handles) + 1,
        frameon=False,
        fontsize=9,
    )

    dataset_label = DATASET_LABELS.get(dataset, dataset)
    figure.suptitle(
        f"Accuracy–Efficiency Pareto Analysis — {dataset_label}",
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Each point is one model–method mean. Accuracy includes zero-scored failures; "
            "efficiency uses records with measured positive cost. Lower cost and higher score "
            "are preferred; outlined labels mark non-dominated points."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.78, hspace=0.34, wspace=0.18)
    return figure, axes


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def output_path_for_dataset(output_dir: Path, dataset: str) -> Path:
    """Return the deterministic PNG path for a dataset Pareto figure."""
    return output_dir / f"{OUTPUT_STEM}_{dataset}.png"


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Generate separate BIRD and Spider accuracy-efficiency Pareto figures"
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
    """Load scored results and generate one six-panel Pareto figure per dataset."""
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
            "input_tokens",
            "total_time_s",
            *METRIC_SOURCE_COLUMNS.values(),
        ),
        datasets=selected_datasets,
        pipelines=("direct", "text_to_sql"),
        models=args.model,
    )
    points = calculate_accuracy_efficiency_points(results)
    if points.empty:
        parser.error("no accuracy-efficiency records matched the selected filters")

    generated: list[tuple[Figure, Path]] = []
    for dataset in selected_datasets:
        if not points["dataset"].eq(dataset).any():
            print(f"Skipping {dataset}: no complete accuracy-efficiency points")
            continue
        figure, _ = plot_accuracy_efficiency_pareto(points, dataset=dataset)
        output_path = output_path_for_dataset(args.output_dir, dataset)
        save_figure(figure, output_path)
        generated.append((figure, output_path))
        print(f"Saved {DATASET_LABELS.get(dataset, dataset)} Pareto figure to {output_path}")
        _print_frontier_summary(points, dataset=dataset)

    if not generated:
        parser.error("no selected dataset had complete accuracy-efficiency points")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        for figure, _ in generated:
            plt.close(figure)
    return 0


def model_family(model_id: str) -> str:
    """Return the model family encoded by a benchmark model identifier."""
    lowered = model_id.lower()
    if lowered.startswith("ministral"):
        return "Ministral"
    if lowered.startswith("qwen"):
        return "Qwen"
    if lowered.startswith("gemma"):
        return "Gemma"
    return "Other"


def _highlight_frontier(
    axis: Axes,
    frontier: pd.DataFrame,
    *,
    cost_column: str,
    score_column: str,
    palette: dict[str, tuple[float, float, float]],
) -> None:
    if frontier.empty:
        return
    label_offsets = _frontier_label_offsets(len(frontier))
    for (_, point), offset_y in zip(frontier.iterrows(), label_offsets, strict=True):
        family = point["model_family"]
        method = point["method"]
        axis.scatter(
            [point[cost_column]],
            [point[score_column]],
            color=[palette[method]],
            marker=FAMILY_MARKERS[family],
            s=105,
            edgecolors="#111111",
            linewidths=1.5,
            zorder=5,
        )
        vertical_alignment = "bottom" if offset_y > 0 else ("top" if offset_y < 0 else "center")
        axis.annotate(
            f"{model_display_name(point['model_id'])}\n{METHOD_LABELS.get(method, method)}",
            xy=(point[cost_column], point[score_column]),
            xytext=(7, offset_y),
            textcoords="offset points",
            ha="left",
            va=vertical_alignment,
            fontsize=6.7,
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.6,
                "shrinkA": 1,
                "shrinkB": 4,
            },
            annotation_clip=False,
            zorder=6,
        )


def _frontier_label_offsets(count: int) -> list[float]:
    if count <= 1:
        return [10.0] * count
    spacing = 24.0 if count <= 3 else 22.0
    midpoint = (count - 1) / 2
    offsets = [(midpoint - index) * spacing for index in range(count)]
    if count == 2:
        return [20.0, -20.0]
    return offsets


def _print_frontier_summary(points: pd.DataFrame, *, dataset: str) -> None:
    panel = points[points["dataset"].eq(dataset)]
    for cost_column, _ in EFFICIENCY_AXES:
        for metric in PARETO_METRICS:
            frontier = pareto_frontier(
                panel,
                cost_column=cost_column,
                score_column=metric,
            )
            names = ", ".join(
                f"{model_display_name(row.model_id)} / {METHOD_LABELS.get(row.method, row.method)}"
                for row in frontier.itertuples(index=False)
            )
            print(f"  {cost_column} vs {metric}: {names}")


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


def _format_tokens(value: float, _position: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def _format_time(value: float, _position: float) -> str:
    return f"{value:g}s"


if __name__ == "__main__":
    raise SystemExit(main())
