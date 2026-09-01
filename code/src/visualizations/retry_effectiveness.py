"""Visualize whether the Text-to-SQL retry mechanism recovers useful answers.

The left panel counts the attempt on which each run first executed successfully, split by whether
its result set matched the gold answer. The right panel reports, for every attempt, the execution
success rate among runs that reached it and the correctness rate among successful executions.
Runs that never execute successfully are shown separately.

    uv run --extra visualizations python -m visualizations.retry_effectiveness
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from config.paths import ProjectPaths
from visualizations.data import DATASET_LABELS, DATASET_ORDER, load_scored_results

OUTPUT_FILENAME = "retry_effectiveness.png"


@dataclass(frozen=True, slots=True)
class RetryEffectiveness:
    """Per-attempt retry statistics plus terminal failure totals."""

    attempts: pd.DataFrame
    total_runs: int
    final_failure_count: int


def calculate_retry_effectiveness(results: pd.DataFrame) -> RetryEffectiveness:
    """Aggregate execution recovery and result correctness by resolving attempt."""
    required_columns = {
        "pipeline",
        "retry_count",
        "execution_success",
        "execution.result_match",
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    data = results[results["pipeline"].eq("text_to_sql")].copy()
    if data.empty:
        return RetryEffectiveness(_empty_attempts(), total_runs=0, final_failure_count=0)

    retry_count = pd.to_numeric(data["retry_count"], errors="coerce")
    if retry_count.isna().any() or (retry_count < 0).any() or (retry_count % 1 != 0).any():
        raise ValueError("retry_count must contain non-negative integers")
    data["retry_count"] = retry_count.astype(int)

    if data["execution_success"].isna().any():
        raise ValueError("execution_success must be available for every Text-to-SQL record")
    data["execution_success"] = data["execution_success"].astype(bool)
    successful = data["execution_success"]

    result_match = data["execution.result_match"]
    if result_match[successful].isna().any():
        raise ValueError("result_match must be available for every successful execution")
    data["result_match"] = result_match.fillna(False).astype(bool)

    max_attempt = int(data["retry_count"].max()) + 1
    rows: list[dict[str, int | float | str]] = []
    for attempt_number in range(1, max_attempt + 1):
        retry_index = attempt_number - 1
        reached = data["retry_count"].ge(retry_index)
        resolved = successful & data["retry_count"].eq(retry_index)
        success_count = int(resolved.sum())
        correct_count = int((resolved & data["result_match"]).sum())
        incorrect_count = success_count - correct_count
        reached_count = int(reached.sum())
        rows.append(
            {
                "attempt_number": attempt_number,
                "attempt_label": _attempt_label(attempt_number),
                "reached_count": reached_count,
                "execution_success_count": success_count,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "execution_recovery_rate": (
                    success_count / reached_count if reached_count else float("nan")
                ),
                "correct_among_success_rate": (
                    correct_count / success_count if success_count else float("nan")
                ),
            }
        )

    attempts = pd.DataFrame.from_records(rows)
    final_failure_count = int((~successful).sum())
    total_runs = len(data)
    if int(attempts["execution_success_count"].sum()) + final_failure_count != total_runs:
        raise ValueError("Retry outcomes do not account for every Text-to-SQL record")
    return RetryEffectiveness(
        attempts,
        total_runs=total_runs,
        final_failure_count=final_failure_count,
    )


def plot_retry_effectiveness(
    analysis: RetryEffectiveness, *, datasets: Sequence[str] = DATASET_ORDER
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create count and rate panels describing retry effectiveness."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter, LogLocator, PercentFormatter

    attempts = analysis.attempts
    required_columns = {
        "attempt_number",
        "attempt_label",
        "reached_count",
        "execution_success_count",
        "correct_count",
        "incorrect_count",
        "execution_recovery_rate",
        "correct_among_success_rate",
    }
    missing = required_columns.difference(attempts.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Retry summary is missing required column(s): {names}")
    if attempts.empty or analysis.total_runs <= 0:
        raise ValueError("No Text-to-SQL retry outcomes are available to plot")

    sns.set_theme(style="whitegrid", context="talk")
    figure, (count_axis, rate_axis) = plt.subplots(
        1, 2, figsize=(15.0, 7.2), gridspec_kw={"width_ratios": (1.15, 1.0)}
    )
    palette = sns.color_palette("colorblind")
    correct_color = palette[2]
    incorrect_color = palette[1]
    failure_color = palette[7]
    recovery_color = palette[0]

    x_positions = list(range(len(attempts)))
    width = 0.34
    correct_counts = attempts["correct_count"].astype(int).tolist()
    incorrect_counts = attempts["incorrect_count"].astype(int).tolist()
    correct_bars = count_axis.bar(
        [position - width / 2 for position in x_positions],
        correct_counts,
        width=width,
        color=correct_color,
        label="Correct result",
    )
    incorrect_bars = count_axis.bar(
        [position + width / 2 for position in x_positions],
        incorrect_counts,
        width=width,
        color=incorrect_color,
        label="Executed, wrong result",
    )

    failure_position = len(attempts)
    failure_bar = count_axis.bar(
        failure_position,
        analysis.final_failure_count,
        width=0.55,
        color=failure_color,
        label="No successful execution",
    )
    _label_log_bars(count_axis, correct_bars, correct_counts)
    _label_log_bars(count_axis, incorrect_bars, incorrect_counts)
    _label_log_bars(count_axis, failure_bar, [analysis.final_failure_count])

    maximum_count = max(
        [analysis.final_failure_count, *correct_counts, *incorrect_counts, 1]
    )
    for position, success_count, correct_count, incorrect_count in zip(
        x_positions,
        attempts["execution_success_count"],
        correct_counts,
        incorrect_counts,
        strict=True,
    ):
        if success_count:
            count_axis.text(
                position,
                max(correct_count, incorrect_count, 1) * 1.9,
                f"Executed: {int(success_count):,}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                weight="bold",
            )

    count_axis.set_yscale("log")
    count_axis.set_ylim(1, maximum_count * 4.2)
    count_axis.yaxis.set_major_locator(LogLocator(base=10))
    count_axis.yaxis.set_major_formatter(FuncFormatter(_format_count_tick))
    count_axis.set_xticks(
        [*x_positions, failure_position],
        [*attempts["attempt_label"], "Never\nexecuted"],
    )
    count_axis.set_title("Outcomes by Resolving Attempt", weight="bold")
    count_axis.set_xlabel("First successful SQL execution")
    count_axis.set_ylabel("Model-question runs (log scale)")
    count_axis.grid(axis="x", visible=False)
    count_axis.legend(frameon=False, fontsize=9, loc="upper right")
    sns.despine(ax=count_axis)

    recovery_rates = attempts["execution_recovery_rate"].mul(100)
    correctness_rates = attempts["correct_among_success_rate"].mul(100)
    recovery_bars = rate_axis.bar(
        [position - width / 2 for position in x_positions],
        recovery_rates,
        width=width,
        color=recovery_color,
        label="Executed / reached attempt",
    )
    correctness_bars = rate_axis.bar(
        [position + width / 2 for position in x_positions],
        correctness_rates,
        width=width,
        color=correct_color,
        label="Correct / successful executions",
    )
    _label_rate_bars(rate_axis, recovery_bars, recovery_rates)
    _label_rate_bars(rate_axis, correctness_bars, correctness_rates)

    rate_tick_labels = [
        f"{label}\nn={int(reached):,}"
        for label, reached in zip(
            attempts["attempt_label"], attempts["reached_count"], strict=True
        )
    ]
    rate_axis.set_ylim(0, 112)
    rate_axis.set_xticks(x_positions, rate_tick_labels)
    rate_axis.set_title("Effectiveness at Each Attempt", weight="bold")
    rate_axis.set_xlabel("Attempt (n = runs reaching that attempt)")
    rate_axis.set_ylabel("Rate (%)")
    rate_axis.yaxis.set_major_formatter(PercentFormatter())
    rate_axis.grid(axis="x", visible=False)
    rate_axis.legend(frameon=False, fontsize=9, loc="upper right")
    sns.despine(ax=rate_axis)

    retried_count = int(attempts.loc[attempts["attempt_number"].eq(2), "reached_count"].sum())
    recovered = int(
        attempts.loc[attempts["attempt_number"].gt(1), "execution_success_count"].sum()
    )
    recovered_correct = int(
        attempts.loc[attempts["attempt_number"].gt(1), "correct_count"].sum()
    )
    dataset_names = " and ".join(DATASET_LABELS.get(value, value) for value in datasets)
    recovery_summary = _recovery_summary(
        retried_count=retried_count,
        recovered=recovered,
        recovered_correct=recovered_correct,
    )
    figure.suptitle("Text-to-SQL Retry Effectiveness", weight="bold", y=0.99)
    figure.text(
        0.5,
        0.012,
        (
            f"{dataset_names} combined; {analysis.total_runs:,} model-question runs. "
            f"{recovery_summary} Correct means exact result-set match against gold."
        ),
        ha="center",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 0.95))
    return figure, (count_axis, rate_axis)


def save_figure(figure: Figure, output_path: Path) -> None:
    """Save a figure, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Visualize Text-to-SQL execution recovery and correctness by retry"
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
    """Load Text-to-SQL results, aggregate retry outcomes, and save the visualization."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    selected_datasets = _canonical_selection(DATASET_ORDER, args.dataset)
    results = load_scored_results(
        args.results,
        columns=(
            "pipeline",
            "retry_count",
            "execution_success",
            "execution.result_match",
        ),
        datasets=selected_datasets,
        pipelines=("text_to_sql",),
        models=args.model,
    )
    analysis = calculate_retry_effectiveness(results)
    if analysis.total_runs == 0:
        parser.error("no Text-to-SQL result records matched the selected filters")

    figure, _ = plot_retry_effectiveness(analysis, datasets=selected_datasets)
    save_figure(figure, args.output)
    print(analysis.attempts.to_string(index=False))
    print(f"Final failures: {analysis.final_failure_count:,}/{analysis.total_runs:,}")
    print(f"Saved figure to {args.output}")

    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


def _label_log_bars(axis: Axes, bars: object, values: Sequence[int]) -> None:
    for bar, value in zip(bars, values, strict=True):
        y = max(value * 1.12, 1.15)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def _label_rate_bars(axis: Axes, bars: object, values: pd.Series) -> None:
    for bar, value in zip(bars, values, strict=True):
        if pd.isna(value):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                1.0,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
        )


def _recovery_summary(*, retried_count: int, recovered: int, recovered_correct: int) -> str:
    if retried_count == 0:
        return "No runs reached a retry."
    recovery_rate = recovered / retried_count
    correct_rate = recovered_correct / recovered if recovered else 0.0
    return (
        f"Retries recovered {recovered:,}/{retried_count:,} runs ({recovery_rate:.1%}); "
        f"{recovered_correct:,} recovered results ({correct_rate:.1%}) were correct."
    )


def _attempt_label(attempt_number: int) -> str:
    return "Initial\nattempt" if attempt_number == 1 else f"Retry {attempt_number - 1}"


def _format_count_tick(value: float, _position: float) -> str:
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


def _empty_attempts() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "attempt_number",
            "attempt_label",
            "reached_count",
            "execution_success_count",
            "correct_count",
            "incorrect_count",
            "execution_recovery_rate",
            "correct_among_success_rate",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
