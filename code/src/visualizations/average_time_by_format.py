"""Plot average end-to-end result time as stacked phases for each format and SQL execution.

The script uses a common complete-case cohort: within each dataset, a question is included only
when every selected model succeeds with every displayed method. Each bar then splits elapsed
time into non-overlapping phases:

* ingestion: request submission until the first output token;
* thinking: first output token until the end of the model's ``<think>`` block;
* generation: remaining model time after ingestion and thinking;
* SQL execution: query execution time (Text-to-SQL only).

For Text-to-SQL retries, ingestion and thinking are summed across every model attempt, just as
``generation_time_s`` is. The script defaults to completed answers and writes
``analysis/figures/average_time_by_format.png``.

    uv run --extra visualizations python -m visualizations.average_time_by_format
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from config.paths import ProjectPaths
from serialization.types import FORMATS
from visualizations.data import (
    HAS_ANSWER_COLUMN,
    METHOD_LABELS,
    METHOD_ORDER,
    SQL_EXECUTION_LABEL,
    load_scored_results,
    select_complete_question_cohort,
)

OUTPUT_FILENAME = "average_time_by_format.png"
ATTEMPT_TTFT_COLUMN = "attempts.llm_metrics.time_to_first_token_s"
ATTEMPT_REASONING_COLUMN = "attempts.llm_metrics.reasoning_time_s"
STACK_COLUMNS: tuple[str, ...] = (
    "ingestion_time_s",
    "thinking_time_s",
    "generation_time_s",
    "sql_execution_time_s",
)
STACK_LABELS: dict[str, str] = {
    "ingestion_time_s": "Ingestion",
    "thinking_time_s": "Thinking",
    "generation_time_s": "Generation",
    "sql_execution_time_s": "SQL execution",
}
BAR_WIDTH = 0.78


def calculate_average_times(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean non-overlapping timing phases for each format/method."""
    required_columns = {
        "pipeline",
        "format",
        "time_to_first_token_s",
        "reasoning_time_s",
        "generation_time_s",
        "query_execution_time_s",
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    data = results.copy()
    data["method"] = data["format"]
    is_sql = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql, "method"] = SQL_EXECUTION_LABEL

    for column in (
        "time_to_first_token_s",
        "reasoning_time_s",
        "generation_time_s",
        "query_execution_time_s",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["method", "generation_time_s"])
    if data.empty:
        return _empty_summary()

    data["ingestion_time_s"] = data["time_to_first_token_s"].fillna(0.0)
    data["thinking_time_s"] = data["reasoning_time_s"].fillna(0.0)

    if ATTEMPT_TTFT_COLUMN in data.columns:
        attempt_ingestion = data[ATTEMPT_TTFT_COLUMN].map(_sum_attempt_metric)
        has_attempt_ingestion = is_sql & attempt_ingestion.notna()
        data.loc[has_attempt_ingestion, "ingestion_time_s"] = attempt_ingestion
    if ATTEMPT_REASONING_COLUMN in data.columns:
        attempt_thinking = data[ATTEMPT_REASONING_COLUMN].map(_sum_attempt_metric)
        has_attempt_thinking = is_sql & attempt_thinking.notna()
        data.loc[has_attempt_thinking, "thinking_time_s"] = attempt_thinking

    model_residual = (
        data["generation_time_s"] - data["ingestion_time_s"] - data["thinking_time_s"]
    )
    # Tiny negatives can occur from independently sampled clocks. Material negatives indicate
    # inconsistent source data and are rejected rather than producing an invalid stacked bar.
    tolerance_s = 1e-6
    invalid = model_residual < -tolerance_s
    if invalid.any():
        count = int(invalid.sum())
        raise ValueError(f"{count} row(s) have timing phases longer than generation_time_s")
    data["generation_phase_time_s"] = model_residual.clip(lower=0.0)
    data["sql_execution_time_s"] = data["query_execution_time_s"].fillna(0.0)
    data.loc[~is_sql, "sql_execution_time_s"] = 0.0

    summary = data.groupby("method", as_index=False, observed=True).agg(
        ingestion_time_s=("ingestion_time_s", "mean"),
        thinking_time_s=("thinking_time_s", "mean"),
        generation_time_s=("generation_phase_time_s", "mean"),
        sql_execution_time_s=("sql_execution_time_s", "mean"),
        answer_count=("generation_phase_time_s", "size"),
    )
    summary["total_time_s"] = summary.loc[:, STACK_COLUMNS].sum(axis=1)
    return _order_methods(summary)


def plot_average_times(
    summary: pd.DataFrame,
    *,
    cohort_question_count: int | None = None,
    cohort_observation_count: int | None = None,
) -> tuple[Figure, Axes]:
    """Create a stacked bar chart from an aggregated timing summary."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import StrMethodFormatter

    required_columns = {"method", "answer_count", "total_time_s", *STACK_COLUMNS}
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Timing summary is missing required column(s): {names}")
    if summary.empty:
        raise ValueError("No completed result timings are available to plot")

    sns.set_theme(style="whitegrid", context="talk")
    figure_width = max(10.5, len(summary) * 1.5)
    figure, axes = plt.subplots(figsize=(figure_width, 7.0))
    x_positions = list(range(len(summary)))
    bottoms = pd.Series(0.0, index=summary.index)
    colors = sns.color_palette("colorblind", n_colors=len(STACK_COLUMNS))

    for column, color in zip(STACK_COLUMNS, colors, strict=True):
        values = summary[column]
        axes.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=STACK_LABELS[column],
            color=color,
            width=BAR_WIDTH,
            edgecolor="white",
            linewidth=0.8,
        )
        bottoms = bottoms + values

    maximum = float(summary["total_time_s"].max())
    label_padding = max(maximum * 0.015, 0.08)
    _add_phase_labels(axes, summary, colors, maximum=maximum)
    for position, total in zip(x_positions, summary["total_time_s"], strict=True):
        axes.text(
            position,
            total + label_padding,
            f"Total {total:,.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    sql_rows = summary.index[summary["method"].eq(SQL_EXECUTION_LABEL)].tolist()
    if sql_rows:
        _add_sql_details_callout(axes, summary, sql_rows[0], colors, maximum=maximum)

    axes.set_title("Average Time to Result by Method", pad=18, weight="bold")
    axes.set_xlabel("Method")
    axes.set_ylabel("Average time (seconds)")
    axes.set_xticks(
        x_positions,
        [METHOD_LABELS.get(method, method) for method in summary["method"]],
    )
    axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes.grid(axis="x", visible=False)
    axes.legend(title="Phase", frameon=False, ncols=2)
    axes.set_ylim(top=maximum + label_padding * 5)
    if sql_rows:
        axes.set_xlim(-0.72, len(summary) + 0.85)
    sns.despine(ax=axes)

    answer_count = int(summary["answer_count"].sum())
    if cohort_question_count is None:
        cohort_description = (
            f"Based on {answer_count:,} completed model-question-method observations."
        )
    else:
        observation_count = cohort_observation_count or answer_count
        cohort_description = (
            f"Common complete-case cohort: {cohort_question_count:,} dataset-question cases; "
            f"{observation_count:,} model-question-method observations."
        )
    figure.text(
        0.5,
        0.01,
        f"{cohort_description} Values are means per result.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure, axes


def _add_phase_labels(
    axes: Axes,
    summary: pd.DataFrame,
    colors: list[tuple[float, float, float]],
    *,
    maximum: float,
) -> None:
    """Label each direct-format stack section without obscuring very thin phases."""
    thin_threshold = maximum * 0.035
    for row_index, row in summary.iterrows():
        if row["method"] == SQL_EXECUTION_LABEL:
            continue
        bottom = 0.0
        for phase_index, column in enumerate(STACK_COLUMNS[:-1]):
            duration = float(row[column])
            midpoint = bottom + duration / 2
            label = _format_duration(duration)
            if duration >= thin_threshold:
                axes.text(
                    row_index,
                    midpoint,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    weight="bold",
                )
            else:
                axes.text(
                    row_index,
                    midpoint,
                    label,
                    ha="center",
                    va="center",
                    color=colors[phase_index],
                    fontsize=8,
                    weight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": colors[phase_index],
                        "linewidth": 0.8,
                        "alpha": 0.95,
                    },
                    zorder=5,
                )
            bottom += duration


def _add_sql_details_callout(
    axes: Axes,
    summary: pd.DataFrame,
    sql_index: int,
    colors: list[tuple[float, float, float]],
    *,
    maximum: float,
) -> None:
    """Annotate all SQL phases in a compact box joined by a short elbow leader."""
    row = summary.loc[sql_index]
    sql_time = float(row["sql_execution_time_s"])
    sql_total = float(row["total_time_s"])
    details = "\n".join(
        (
            f"Ingestion   {_format_duration(float(row['ingestion_time_s']))}",
            f"Thinking    {_format_duration(float(row['thinking_time_s']))}",
            f"Generation  {_format_duration(float(row['generation_time_s']))}",
            f"SQL query   {sql_time:.3f}s",
        )
    )
    axes.annotate(
        details,
        xy=(sql_index + BAR_WIDTH / 2, sql_total - sql_time / 2),
        xytext=(sql_index + 0.66, sql_total + maximum * 0.035),
        ha="left",
        va="center",
        fontsize=8.5,
        linespacing=1.35,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": colors[-1],
            "linewidth": 1.4,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": colors[-1],
            "linewidth": 1.4,
            "connectionstyle": "angle,angleA=180,angleB=90,rad=5",
        },
        zorder=6,
    )


def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s" if seconds >= 0.01 else f"{seconds:.3f}s"


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Plot average result time split into ingestion, thinking, generation, and SQL"
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
        default=None,
        help="dataset to include; repeat to include multiple datasets (default: all)",
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        default=None,
        help=(
            "pipeline to include; repeat for multiple pipelines "
            "(default: direct and text_to_sql)"
        ),
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
    """Load scored results, aggregate timing phases, and save the visualization."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    pipelines = args.pipeline if args.pipeline is not None else ["direct", "text_to_sql"]

    results = load_scored_results(
        args.results,
        columns=(
            "dataset",
            "example_id",
            "model_id",
            "pipeline",
            "format",
            "error",
            HAS_ANSWER_COLUMN,
            "time_to_first_token_s",
            "reasoning_time_s",
            "generation_time_s",
            "query_execution_time_s",
            ATTEMPT_TTFT_COLUMN,
            ATTEMPT_REASONING_COLUMN,
        ),
        datasets=args.dataset,
        pipelines=pipelines,
        formats=args.formats,
        models=args.model,
    )
    cohort = select_complete_question_cohort(results)
    summary = calculate_average_times(cohort.records)
    if summary.empty:
        parser.error("no result timing records matched the selected filters")

    figure, _ = plot_average_times(
        summary,
        cohort_question_count=cohort.question_count,
        cohort_observation_count=cohort.observation_count,
    )
    save_figure(figure, args.output)

    printable = summary.copy()
    for column in (*STACK_COLUMNS, "total_time_s"):
        printable[column] = printable[column].map(lambda value: f"{value:,.4f}")
    print(printable.to_string(index=False))
    print(f"Saved figure to {args.output}")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


def _sum_attempt_metric(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.sum()) if not numeric.empty else None


def _order_methods(summary: pd.DataFrame) -> pd.DataFrame:
    positions = {method: index for index, method in enumerate(METHOD_ORDER)}
    fallback = len(positions)
    return summary.sort_values(
        "method", key=lambda values: values.map(lambda value: positions.get(value, fallback))
    ).reset_index(drop=True)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["method", *STACK_COLUMNS, "answer_count", "total_time_s"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
