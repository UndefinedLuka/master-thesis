"""Plot Time to First Token (TTFT) against input-token count by method.

BIRD and Spider observations share one plot. Both axes are logarithmic because prompt sizes and
TTFT span several orders of magnitude. Points are colored by method and each method receives a
log-log least-squares trend line. Text-to-SQL uses first-attempt input tokens and first-attempt
TTFT so retries do not create mismatched pairs.

    uv run --extra visualizations python -m visualizations.ttft_vs_input_tokens
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

OUTPUT_FILENAME = "ttft_vs_input_tokens.png"
ATTEMPT_INPUT_TOKENS_COLUMN = "attempts.llm_metrics.input_tokens"
ATTEMPT_TTFT_COLUMN = "attempts.llm_metrics.time_to_first_token_s"
POINT_SAMPLE_PER_GROUP = 1_200


def prepare_ttft_data(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize methods and retain valid prompt-size/TTFT observation pairs."""
    required_columns = {
        "dataset",
        "pipeline",
        "format",
        "input_tokens",
        "time_to_first_token_s",
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    data = results.copy()
    data["method"] = data["format"]
    is_sql = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql, "method"] = SQL_EXECUTION_LABEL
    data["input_tokens"] = pd.to_numeric(data["input_tokens"], errors="coerce")
    data["time_to_first_token_s"] = pd.to_numeric(
        data["time_to_first_token_s"], errors="coerce"
    )

    if ATTEMPT_INPUT_TOKENS_COLUMN in data.columns:
        first_attempt_tokens = data[ATTEMPT_INPUT_TOKENS_COLUMN].map(_first_numeric)
        available = is_sql & first_attempt_tokens.notna()
        data.loc[available, "input_tokens"] = first_attempt_tokens
    if ATTEMPT_TTFT_COLUMN in data.columns:
        first_attempt_ttft = data[ATTEMPT_TTFT_COLUMN].map(_first_numeric)
        available = is_sql & first_attempt_ttft.notna()
        data.loc[available, "time_to_first_token_s"] = first_attempt_ttft

    data = data.dropna(subset=["dataset", "method", "input_tokens", "time_to_first_token_s"])
    data = data[(data["input_tokens"] > 0) & (data["time_to_first_token_s"] > 0)].copy()
    data["method"] = pd.Categorical(data["method"], categories=METHOD_ORDER, ordered=True)
    return data.sort_values(["dataset", "method", "input_tokens"]).reset_index(drop=True)


def plot_ttft_vs_input_tokens(data: pd.DataFrame) -> tuple[Figure, Axes]:
    """Create one combined scatterplot with per-method log-log trend lines."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator

    required_columns = {"dataset", "method", "input_tokens", "time_to_first_token_s"}
    missing = required_columns.difference(data.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"TTFT data is missing required column(s): {names}")
    if data.empty:
        raise ValueError("No completed positive token/TTFT observations are available to plot")

    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(data["dataset"])]
    methods = [method for method in METHOD_ORDER if method in set(data["method"].dropna())]
    if not datasets or not methods:
        raise ValueError("No recognized datasets or methods are available to plot")

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(11.5, 7.5))
    colors = sns.color_palette("colorblind", n_colors=len(methods))
    palette = dict(zip(methods, colors, strict=True))
    sampled = _sample_points(data)

    sns.scatterplot(
        data=sampled,
        x="input_tokens",
        y="time_to_first_token_s",
        hue="method",
        hue_order=methods,
        palette=palette,
        s=14,
        alpha=0.22,
        linewidth=0,
        legend=False,
        ax=axis,
    )
    for method in methods:
        method_data = data[data["method"].eq(method)]
        trend = _log_log_trend(method_data)
        if trend is not None:
            trend_x, trend_y = trend
            axis.plot(trend_x, trend_y, color=palette[method], linewidth=2.2, alpha=0.95)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Input tokens")
    axis.set_ylabel("Time to First Token (seconds)")
    axis.xaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_tokens))
    axis.yaxis.set_major_formatter(FuncFormatter(_format_seconds))
    axis.grid(which="major", alpha=0.55)
    axis.grid(which="minor", alpha=0.16)
    sns.despine(ax=axis)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=palette[method],
            marker="o",
            markersize=6,
            linewidth=2,
            label=METHOD_LABELS.get(method, method),
        )
        for method in methods
    ]
    figure.legend(
        handles=legend_handles,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=min(4, len(methods)),
        frameon=False,
    )
    figure.suptitle("TTFT versus Input-Token Count", weight="bold", y=0.99)
    dataset_names = " and ".join(DATASET_LABELS.get(dataset, dataset) for dataset in datasets)
    figure.text(
        0.5,
        0.015,
        (
            f"{dataset_names} observations are combined. Points are a deterministic sample; "
            "lines are per-method log–log trends. Both axes are logarithmic."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.82))
    return figure, axis


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Plot TTFT against input-token count with selected datasets combined"
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
    """Load completed results and save the TTFT/input-token visualization."""
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
            "input_tokens",
            "time_to_first_token_s",
            ATTEMPT_INPUT_TOKENS_COLUMN,
            ATTEMPT_TTFT_COLUMN,
        ),
        datasets=selected_datasets,
        pipelines=("direct", "text_to_sql"),
        models=args.model,
        answers_only=True,
    )
    if args.formats is not None:
        keep_direct = results["format"].isin(args.formats)
        keep_sql = results["pipeline"].eq("text_to_sql")
        results = results[keep_direct | keep_sql]

    data = prepare_ttft_data(results)
    if data.empty:
        parser.error("no completed positive token/TTFT records matched the selected filters")

    figure, _ = plot_ttft_vs_input_tokens(data)
    save_figure(figure, args.output)
    print(_relationship_summary(data).to_string(index=False))
    print(f"Saved figure to {args.output}")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


def _first_numeric(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None


def _sample_points(data: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        group.sample(n=min(len(group), POINT_SAMPLE_PER_GROUP), random_state=42)
        for _, group in data.groupby("method", observed=True)
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else data.iloc[0:0].copy()


def _log_log_trend(data: pd.DataFrame) -> tuple[list[float], list[float]] | None:
    if len(data) < 2:
        return None
    log_x = [math.log10(float(value)) for value in data["input_tokens"]]
    log_y = [math.log10(float(value)) for value in data["time_to_first_token_s"]]
    if len(set(log_x)) < 2:
        return None
    slope, intercept = statistics.linear_regression(log_x, log_y)
    low, high = min(log_x), max(log_x)
    trend_log_x = [low + (high - low) * index / 49 for index in range(50)]
    trend_x = [10**value for value in trend_log_x]
    trend_y = [10 ** (intercept + slope * value) for value in trend_log_x]
    return trend_x, trend_y


def _relationship_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in data.groupby("method", observed=True):
        log_tokens = [math.log10(float(value)) for value in group["input_tokens"]]
        log_ttft = [math.log10(float(value)) for value in group["time_to_first_token_s"]]
        correlation = statistics.correlation(log_tokens, log_ttft) if len(group) > 1 else None
        rows.append(
            {
                "method": method,
                "count": len(group),
                "log_correlation": correlation,
            }
        )
    return pd.DataFrame(rows)


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


def _format_seconds(value: float, _position: float) -> str:
    if value >= 1:
        return f"{value:g}s"
    return f"{value:g}s"


if __name__ == "__main__":
    raise SystemExit(main())
