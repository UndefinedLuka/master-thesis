"""Plot average input-token usage for each serialization format.

The script uses a common complete-case cohort: within each dataset, a question is included only
when every selected model succeeds with every displayed method. This prevents context failures
from giving methods or models unequal, artificially easier question sets.

    uv run --extra visualizations python -m visualizations.average_input_tokens_by_format
    uv run --extra visualizations python -m visualizations.average_input_tokens_by_format \
        --dataset bird --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from config.paths import ProjectPaths
from serialization.types import FORMATS
from visualizations.data import (
    HAS_ANSWER_COLUMN,
    METHOD_LABELS,
    SQL_EXECUTION_LABEL,
    load_scored_results,
    select_complete_question_cohort,
)

OUTPUT_FILENAME = "average_input_tokens_by_format.png"


def calculate_average_input_tokens(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean input tokens by format, including Text-to-SQL as its own category."""
    required_columns = {"pipeline", "format", "input_tokens"}
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    data = results.loc[:, ["pipeline", "format", "input_tokens"]].copy()
    is_sql_execution = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql_execution, "format"] = SQL_EXECUTION_LABEL
    data["input_tokens"] = pd.to_numeric(data["input_tokens"], errors="coerce")
    data = data.dropna(subset=["format", "input_tokens"])
    if data.empty:
        return pd.DataFrame(columns=["format", "average_input_tokens", "answer_count"])

    summary = data.groupby("format", as_index=False, observed=True).agg(
        average_input_tokens=("input_tokens", "mean"),
        answer_count=("input_tokens", "size"),
    )
    summary = summary.sort_values("average_input_tokens", ascending=True).reset_index(drop=True)

    # Preserve the token-based ordering of serialization formats, but keep the baseline method
    # adjacent to XML as requested, irrespective of its much smaller average token count.
    order = summary["format"].tolist()
    if SQL_EXECUTION_LABEL in order:
        order.remove(SQL_EXECUTION_LABEL)
        insertion_index = order.index("xml") + 1 if "xml" in order else len(order)
        order.insert(insertion_index, SQL_EXECUTION_LABEL)
        positions = {category: index for index, category in enumerate(order)}
        summary = summary.sort_values(
            "format", key=lambda values: values.map(positions)
        ).reset_index(drop=True)
    return summary


def plot_average_input_tokens(
    summary: pd.DataFrame,
    *,
    cohort_question_count: int | None = None,
    cohort_observation_count: int | None = None,
) -> tuple[Figure, Axes]:
    """Create a publication-ready bar chart from an aggregated token summary."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import StrMethodFormatter

    required_columns = {"format", "average_input_tokens", "answer_count"}
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Token summary is missing required column(s): {names}")
    if summary.empty:
        raise ValueError("No formatted answers are available to plot")

    sns.set_theme(style="whitegrid", context="talk")
    figure_width = max(9.0, len(summary) * 1.35)
    figure, axes = plt.subplots(figsize=(figure_width, 6.5))
    order = summary["format"].tolist()
    palette = sns.color_palette("colorblind", n_colors=len(order))

    sns.barplot(
        data=summary,
        x="format",
        y="average_input_tokens",
        hue="format",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        ax=axes,
    )

    axes.set_title("Average Input Tokens by Method", pad=18, weight="bold")
    axes.set_xlabel("Method")
    axes.set_ylabel("Average input tokens")
    axes.set_xticks(
        range(len(order)),
        [METHOD_LABELS.get(method, method) for method in order],
    )
    axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes.grid(axis="x", visible=False)
    sns.despine(ax=axes)

    for container in axes.containers:
        axes.bar_label(
            container,
            labels=[f"{bar.get_height():,.0f}" for bar in container],
            padding=4,
            fontsize=10,
        )

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
        cohort_description,
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure, axes


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Plot mean input-token usage for each serialization format"
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
    """Load scored results, aggregate input tokens, and save the visualization."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        # macOS may choose a GUI backend that aborts in headless shells. Saving needs no GUI.
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
            "input_tokens",
        ),
        datasets=args.dataset,
        pipelines=pipelines,
        formats=args.formats,
        models=args.model,
    )
    cohort = select_complete_question_cohort(results)
    summary = calculate_average_input_tokens(cohort.records)
    if summary.empty:
        parser.error("no formatted result records matched the selected filters")

    figure, _ = plot_average_input_tokens(
        summary,
        cohort_question_count=cohort.question_count,
        cohort_observation_count=cohort.observation_count,
    )
    save_figure(figure, args.output)

    printable = summary.copy()
    printable["average_input_tokens"] = printable["average_input_tokens"].map(
        lambda value: f"{value:,.2f}"
    )
    print(printable.to_string(index=False))
    print(f"Saved figure to {args.output}")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
