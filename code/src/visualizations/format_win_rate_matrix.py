"""Generate per-dataset paired format win-rate matrices.

For each accuracy metric, cell ``(A, B)`` is the percentage of paired model-question
observations for which method A scored strictly higher than method B. Pairing is performed on
``(dataset, model_id, example_id)``, ensuring both methods answered the same question with the
same model. Ties remain in the denominator but count as a win for neither method, so opposite
cells need not sum to 100%.

Each output figure contains matrices for normalized exact match, ROUGE-L F1, and BERTScore F1.
Annotations include both win percentage and paired sample size. Missing comparisons are hatched
rather than treated as zero.

    uv run --extra visualizations python -m visualizations.format_win_rate_matrix
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
from visualizations.accuracy_heatmaps import HEATMAP_METRICS
from visualizations.accuracy_table import METRIC_SOURCE_COLUMNS
from visualizations.data import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_TITLES,
    SQL_EXECUTION_LABEL,
    load_scored_results,
)

OUTPUT_STEM = "format_win_rate_matrix"


def calculate_format_win_rates(
    results: pd.DataFrame,
    *,
    datasets: Sequence[str] = DATASET_ORDER,
    methods: Sequence[str] = METHOD_ORDER,
    metrics: Sequence[str] = HEATMAP_METRICS,
) -> pd.DataFrame:
    """Calculate strict paired win rates for every dataset, metric, and method pair."""
    required_columns = {
        "dataset",
        "model_id",
        "example_id",
        "pipeline",
        "format",
        *METRIC_SOURCE_COLUMNS.values(),
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    selected_datasets = tuple(dict.fromkeys(datasets))
    selected_methods = tuple(dict.fromkeys(methods))
    selected_metrics = tuple(dict.fromkeys(metrics))
    if not selected_datasets or not selected_methods or not selected_metrics:
        raise ValueError("At least one dataset, method, and metric must be selected")

    data = results.loc[:, sorted(required_columns)].copy()
    data["method"] = data["format"]
    is_sql = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql, "method"] = SQL_EXECUTION_LABEL
    data = data[
        data["dataset"].isin(selected_datasets) & data["method"].isin(selected_methods)
    ].dropna(subset=["model_id", "example_id"])

    identity_columns = ["dataset", "model_id", "example_id", "method"]
    duplicates = data.duplicated(identity_columns, keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate dataset/model/question/method result rows"
        )

    renamed = data.rename(
        columns={source: metric for metric, source in METRIC_SOURCE_COLUMNS.items()}
    )
    for metric in selected_metrics:
        renamed[metric] = pd.to_numeric(renamed[metric], errors="coerce")

    rows: list[dict[str, int | float | str]] = []
    for dataset in selected_datasets:
        dataset_data = renamed[renamed["dataset"].eq(dataset)]
        for metric in selected_metrics:
            matrix = dataset_data.pivot(
                index=["model_id", "example_id"],
                columns="method",
                values=metric,
            )
            for row_method in selected_methods:
                for column_method in selected_methods:
                    paired = _paired_scores(matrix, row_method, column_method)
                    pair_count = len(paired)
                    win_count = int((paired["score_a"] > paired["score_b"]).sum())
                    loss_count = int((paired["score_a"] < paired["score_b"]).sum())
                    tie_count = pair_count - win_count - loss_count
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "row_method": row_method,
                            "column_method": column_method,
                            "pair_count": pair_count,
                            "win_count": win_count,
                            "loss_count": loss_count,
                            "tie_count": tie_count,
                            "win_rate_pct": (
                                win_count / pair_count * 100 if pair_count else float("nan")
                            ),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def plot_format_win_rate_matrices(
    summary: pd.DataFrame,
    *,
    dataset: str,
    methods: Sequence[str] = METHOD_ORDER,
    metrics: Sequence[str] = HEATMAP_METRICS,
) -> tuple[Figure, list[Axes]]:
    """Create one three-metric paired win-rate figure for a dataset."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    required_columns = {
        "dataset",
        "metric",
        "row_method",
        "column_method",
        "pair_count",
        "win_rate_pct",
    }
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Win-rate summary is missing required column(s): {names}")

    panel = summary[summary["dataset"].eq(dataset)]
    if panel.empty:
        raise ValueError(f"No paired win-rate data are available for dataset: {dataset}")

    selected_methods = tuple(methods)
    selected_metrics = tuple(metrics)
    labels = [METHOD_LABELS.get(method, method) for method in selected_methods]

    sns.set_theme(style="white", context="talk")
    figure, axes_array = plt.subplots(
        1,
        len(selected_metrics),
        figsize=(max(16.5, len(selected_metrics) * 5.7), 7.1),
        squeeze=False,
    )
    axes = list(axes_array[0])
    colorbar_axis = figure.add_axes((0.925, 0.18, 0.016, 0.65))
    color_map = sns.color_palette("viridis", as_cmap=True)

    for index, (axis, metric) in enumerate(zip(axes, selected_metrics, strict=True)):
        metric_panel = panel[panel["metric"].eq(metric)]
        rates = metric_panel.pivot(
            index="row_method", columns="column_method", values="win_rate_pct"
        ).reindex(index=selected_methods, columns=selected_methods)
        pair_counts = metric_panel.pivot(
            index="row_method", columns="column_method", values="pair_count"
        ).reindex(index=selected_methods, columns=selected_methods)
        diagonal_mask = pd.DataFrame(
            [
                [row_method == column_method for column_method in selected_methods]
                for row_method in selected_methods
            ],
            index=selected_methods,
            columns=selected_methods,
        )
        missing_mask = rates.isna()
        plot_mask = diagonal_mask | missing_mask

        sns.heatmap(
            rates,
            mask=plot_mask,
            cmap=color_map,
            vmin=0.0,
            vmax=100.0,
            linewidths=0.8,
            linecolor="white",
            cbar=index == len(selected_metrics) - 1,
            cbar_ax=colorbar_axis if index == len(selected_metrics) - 1 else None,
            ax=axis,
        )
        _annotate_matrix(
            axis,
            rates,
            pair_counts,
            diagonal_mask=diagonal_mask,
            missing_mask=missing_mask,
        )
        axis.set_title(METRIC_TITLES.get(metric, metric), weight="bold", pad=12)
        axis.set_xlabel("Method B (comparison)")
        axis.set_ylabel("Method A (winner)" if index == 0 else "")
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_yticklabels(labels, rotation=0)
        axis.tick_params(axis="both", labelsize=9, labelleft=True)

    colorbar_axis.set_ylabel("A beats B (%)", rotation=270, labelpad=22)
    colorbar_axis.yaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    figure.suptitle(f"Paired Format Win Rates — {dataset_label}", weight="bold", y=0.99)
    figure.text(
        0.5,
        0.015,
        (
            "Cell (A, B) is the percentage of paired model–question observations where A > B; "
            "n is the paired sample size. Ties stay in the denominator and count for neither "
            "method, so opposite cells need not sum to 100%."
        ),
        ha="center",
        fontsize=9.3,
    )
    figure.subplots_adjust(left=0.075, right=0.90, bottom=0.17, top=0.88, wspace=0.24)
    return figure, axes


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def output_path_for_dataset(output_dir: Path, dataset: str) -> Path:
    """Return the deterministic PNG path for a dataset win-rate figure."""
    return output_dir / f"{OUTPUT_STEM}_{dataset}.png"


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Generate separate BIRD and Spider paired format win-rate matrices"
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
    """Load scored results and generate one paired win-rate figure per dataset."""
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
            "model_id",
            "example_id",
            "pipeline",
            "format",
            *METRIC_SOURCE_COLUMNS.values(),
        ),
        datasets=selected_datasets,
        pipelines=("direct", "text_to_sql"),
        models=args.model,
    )
    summary = calculate_format_win_rates(results, datasets=selected_datasets)
    if summary.empty:
        parser.error("no paired accuracy records matched the selected filters")

    generated: list[tuple[Figure, Path]] = []
    for dataset in selected_datasets:
        if not summary["dataset"].eq(dataset).any():
            print(f"Skipping {dataset}: no paired accuracy records")
            continue
        figure, _ = plot_format_win_rate_matrices(summary, dataset=dataset)
        output_path = output_path_for_dataset(args.output_dir, dataset)
        save_figure(figure, output_path)
        generated.append((figure, output_path))
        print(f"Saved {DATASET_LABELS.get(dataset, dataset)} win-rate matrices to {output_path}")
        _print_coverage(summary, dataset=dataset)

    if not generated:
        parser.error("no selected dataset had paired accuracy records")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        for figure, _ in generated:
            plt.close(figure)
    return 0


def _paired_scores(matrix: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    if method_a not in matrix.columns or method_b not in matrix.columns:
        return pd.DataFrame(columns=("score_a", "score_b"))
    if method_a == method_b:
        values = matrix[method_a].dropna()
        return pd.DataFrame({"score_a": values, "score_b": values})
    return matrix[[method_a, method_b]].dropna().rename(
        columns={method_a: "score_a", method_b: "score_b"}
    )


def _annotate_matrix(
    axis: Axes,
    rates: pd.DataFrame,
    pair_counts: pd.DataFrame,
    *,
    diagonal_mask: pd.DataFrame,
    missing_mask: pd.DataFrame,
) -> None:
    from matplotlib.patches import Rectangle

    for row_index in range(rates.shape[0]):
        for column_index in range(rates.shape[1]):
            if diagonal_mask.iat[row_index, column_index]:
                axis.add_patch(
                    Rectangle(
                        (column_index, row_index),
                        1,
                        1,
                        facecolor="#e4e4e4",
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                axis.text(
                    column_index + 0.5,
                    row_index + 0.5,
                    "—",
                    ha="center",
                    va="center",
                    color="#666666",
                    weight="bold",
                )
                continue
            if missing_mask.iat[row_index, column_index]:
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
                    color="#555555",
                    fontsize=8,
                    weight="bold",
                )
                continue

            rate = float(rates.iat[row_index, column_index])
            pair_count = int(pair_counts.iat[row_index, column_index])
            text_color = "white" if rate < 48 else "#222222"
            axis.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{rate:.1f}%\nn={pair_count:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.4,
                weight="bold",
                linespacing=1.25,
            )


def _print_coverage(summary: pd.DataFrame, *, dataset: str) -> None:
    panel = summary[
        summary["dataset"].eq(dataset)
        & summary["row_method"].ne(summary["column_method"])
        & summary["pair_count"].gt(0)
    ]
    if panel.empty:
        print("  No off-diagonal paired comparisons")
        return
    minimum = int(panel["pair_count"].min())
    maximum = int(panel["pair_count"].max())
    print(f"  Paired sample sizes range from {minimum:,} to {maximum:,}")


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


if __name__ == "__main__":
    raise SystemExit(main())
