"""Generate every analysis figure and LaTeX table from one result snapshot.

The command loads the scored NDJSON tree once, then runs every visualization against that same
in-memory snapshot. This is both faster than invoking each script separately and prevents figures
from observing different record counts while benchmark files are still being appended.

    uv run --extra visualizations thesis-visualize-all
    uv run --extra visualizations thesis-visualize-all --dataset bird
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from config.paths import ProjectPaths
from visualizations import accuracy_by_model_size as model_size_plot
from visualizations import accuracy_efficiency_pareto as pareto_plot
from visualizations import accuracy_heatmaps as heatmap_plot
from visualizations import accuracy_table
from visualizations import average_input_tokens_by_format as token_plot
from visualizations import average_time_by_format as time_plot
from visualizations import context_exceeded_by_format as context_plot
from visualizations import format_win_rate_matrix as win_rate_plot
from visualizations import latency_distribution_by_format as latency_plot
from visualizations import retry_effectiveness as retry_plot
from visualizations import ttft_vs_input_tokens as ttft_plot
from visualizations.data import (
    DATASET_LABELS,
    DATASET_ORDER,
    HAS_ANSWER_COLUMN,
    load_scored_results,
    select_complete_question_cohort,
)

ALL_COLUMNS: tuple[str, ...] = (
    "dataset",
    "pipeline",
    "format",
    "model_id",
    "example_id",
    "input_tokens",
    "time_to_first_token_s",
    "reasoning_time_s",
    "generation_time_s",
    "query_execution_time_s",
    "total_time_s",
    "retry_count",
    "execution_success",
    "error",
    HAS_ANSWER_COLUMN,
    "execution.result_match",
    time_plot.ATTEMPT_TTFT_COLUMN,
    time_plot.ATTEMPT_REASONING_COLUMN,
    ttft_plot.ATTEMPT_INPUT_TOKENS_COLUMN,
    *accuracy_table.METRIC_SOURCE_COLUMNS.values(),
)


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Shared result snapshot and output configuration for all generator stages."""

    records: pd.DataFrame
    completed_answers: pd.DataFrame
    datasets: tuple[str, ...]
    figures_dir: Path
    tables_dir: Path


@dataclass(frozen=True, slots=True)
class GenerationStage:
    """One named generator used by the all-visualizations command."""

    name: str
    run: Callable[[GenerationContext], list[Path]]


def generate_all(context: GenerationContext) -> list[Path]:
    """Run all generators and return every produced artifact path."""
    outputs: list[Path] = []
    total = len(GENERATION_STAGES)
    for index, stage in enumerate(GENERATION_STAGES, start=1):
        print(f"[{index}/{total}] {stage.name} ...")
        try:
            stage_outputs = stage.run(context)
        except Exception as exc:
            raise RuntimeError(f"Generation stage failed: {stage.name}") from exc
        outputs.extend(stage_outputs)
        for output in stage_outputs:
            print(f"  -> {output}")
    return outputs


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Generate every visualization and analysis table from one result snapshot"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=project_paths.results_dir,
        help="scored NDJSON file or result directory (default: results/)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=project_paths.root / "analysis" / "figures",
        help="PNG output directory (default: analysis/figures/)",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=project_paths.root / "analysis" / "tables",
        help="LaTeX table output directory (default: analysis/tables/)",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load one result snapshot and generate every analysis artifact."""
    parser = build_parser()
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    selected_datasets = _canonical_selection(DATASET_ORDER, args.dataset)
    print(
        "Loading one scored-result snapshot for "
        + ", ".join(DATASET_LABELS.get(dataset, dataset) for dataset in selected_datasets)
        + " ..."
    )
    records = load_scored_results(
        args.results,
        columns=ALL_COLUMNS,
        datasets=selected_datasets,
        pipelines=("direct", "text_to_sql"),
        models=args.model,
    )
    if records.empty:
        parser.error("no scored result records matched the selected filters")

    completed_answers = records[
        records["error"].isna() & records[HAS_ANSWER_COLUMN].eq(True)
    ].copy()
    context = GenerationContext(
        records=records,
        completed_answers=completed_answers,
        datasets=selected_datasets,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )
    outputs = generate_all(context)
    print(
        f"Generated {len(outputs)} artifacts from {len(records):,} result records "
        f"({len(completed_answers):,} completed answers)."
    )
    return 0


def _generate_average_input_tokens(context: GenerationContext) -> list[Path]:
    cohort = select_complete_question_cohort(context.records)
    summary = token_plot.calculate_average_input_tokens(cohort.records)
    figure, _ = token_plot.plot_average_input_tokens(
        summary,
        cohort_question_count=cohort.question_count,
        cohort_observation_count=cohort.observation_count,
    )
    path = context.figures_dir / token_plot.OUTPUT_FILENAME
    return [_save_and_close(figure, token_plot.save_figure, path)]


def _generate_average_time(context: GenerationContext) -> list[Path]:
    cohort = select_complete_question_cohort(context.records)
    summary = time_plot.calculate_average_times(cohort.records)
    figure, _ = time_plot.plot_average_times(
        summary,
        cohort_question_count=cohort.question_count,
        cohort_observation_count=cohort.observation_count,
    )
    path = context.figures_dir / time_plot.OUTPUT_FILENAME
    return [_save_and_close(figure, time_plot.save_figure, path)]


def _generate_context_exceeded(context: GenerationContext) -> list[Path]:
    summary = context_plot.calculate_context_exceeded_counts(
        context.records,
        datasets=context.datasets,
    )
    figure, _ = context_plot.plot_context_exceeded_counts(summary)
    path = context.figures_dir / context_plot.OUTPUT_FILENAME
    return [_save_and_close(figure, context_plot.save_figure, path)]


def _generate_latency_distribution(context: GenerationContext) -> list[Path]:
    data = latency_plot.prepare_latency_data(context.completed_answers)
    figure, _ = latency_plot.plot_latency_distribution(data)
    path = context.figures_dir / latency_plot.OUTPUT_FILENAME
    return [_save_and_close(figure, latency_plot.save_figure, path)]


def _generate_ttft_relationship(context: GenerationContext) -> list[Path]:
    data = ttft_plot.prepare_ttft_data(context.completed_answers)
    figure, _ = ttft_plot.plot_ttft_vs_input_tokens(data)
    path = context.figures_dir / ttft_plot.OUTPUT_FILENAME
    return [_save_and_close(figure, ttft_plot.save_figure, path)]


def _generate_retry_effectiveness(context: GenerationContext) -> list[Path]:
    analysis = retry_plot.calculate_retry_effectiveness(context.records)
    figure, _ = retry_plot.plot_retry_effectiveness(analysis, datasets=context.datasets)
    path = context.figures_dir / retry_plot.OUTPUT_FILENAME
    return [_save_and_close(figure, retry_plot.save_figure, path)]


def _generate_accuracy_table(context: GenerationContext) -> list[Path]:
    tables = accuracy_table.render_dataset_accuracy_tables(
        context.records,
        datasets=context.datasets,
    )
    base_path = context.tables_dir / accuracy_table.OUTPUT_FILENAME
    outputs: list[Path] = []
    for dataset, latex in tables.items():
        path = accuracy_table.output_path_for_dataset(base_path, dataset)
        accuracy_table.write_latex_table(path, latex)
        outputs.append(path)
    return outputs


def _generate_accuracy_heatmaps(context: GenerationContext) -> list[Path]:
    summary = heatmap_plot.calculate_dataset_accuracy_means(context.records)
    outputs: list[Path] = []
    for dataset in context.datasets:
        if not summary["dataset"].eq(dataset).any():
            continue
        figure, _ = heatmap_plot.plot_accuracy_heatmaps(summary, dataset=dataset)
        path = heatmap_plot.output_path_for_dataset(context.figures_dir, dataset)
        outputs.append(_save_and_close(figure, heatmap_plot.save_figure, path))
    return outputs


def _generate_accuracy_efficiency(context: GenerationContext) -> list[Path]:
    points = pareto_plot.calculate_accuracy_efficiency_points(context.records)
    outputs: list[Path] = []
    for dataset in context.datasets:
        if not points["dataset"].eq(dataset).any():
            continue
        figure, _ = pareto_plot.plot_accuracy_efficiency_pareto(points, dataset=dataset)
        path = pareto_plot.output_path_for_dataset(context.figures_dir, dataset)
        outputs.append(_save_and_close(figure, pareto_plot.save_figure, path))
    return outputs


def _generate_model_size_accuracy(context: GenerationContext) -> list[Path]:
    summary = model_size_plot.calculate_model_size_accuracy(context.records)
    outputs: list[Path] = []
    for dataset in context.datasets:
        if not summary["dataset"].eq(dataset).any():
            continue
        figure, _ = model_size_plot.plot_accuracy_by_model_size(summary, dataset=dataset)
        path = model_size_plot.output_path_for_dataset(context.figures_dir, dataset)
        outputs.append(_save_and_close(figure, model_size_plot.save_figure, path))
    return outputs


def _generate_format_win_rates(context: GenerationContext) -> list[Path]:
    summary = win_rate_plot.calculate_format_win_rates(
        context.records,
        datasets=context.datasets,
    )
    outputs: list[Path] = []
    for dataset in context.datasets:
        if not summary["dataset"].eq(dataset).any():
            continue
        figure, _ = win_rate_plot.plot_format_win_rate_matrices(summary, dataset=dataset)
        path = win_rate_plot.output_path_for_dataset(context.figures_dir, dataset)
        outputs.append(_save_and_close(figure, win_rate_plot.save_figure, path))
    return outputs


def _save_and_close(
    figure: Figure,
    save_figure: Callable[[Figure, Path], None],
    path: Path,
) -> Path:
    save_figure(figure, path)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


GENERATION_STAGES: tuple[GenerationStage, ...] = (
    GenerationStage("Average input-token usage", _generate_average_input_tokens),
    GenerationStage("Average time decomposition", _generate_average_time),
    GenerationStage("Context-exceeded counts", _generate_context_exceeded),
    GenerationStage("Latency distributions", _generate_latency_distribution),
    GenerationStage("TTFT versus input tokens", _generate_ttft_relationship),
    GenerationStage("Text-to-SQL retry effectiveness", _generate_retry_effectiveness),
    GenerationStage("Mean-accuracy LaTeX table", _generate_accuracy_table),
    GenerationStage("Accuracy heatmaps", _generate_accuracy_heatmaps),
    GenerationStage("Accuracy-efficiency Pareto plots", _generate_accuracy_efficiency),
    GenerationStage("Accuracy by model size", _generate_model_size_accuracy),
    GenerationStage("Paired format win-rate matrices", _generate_format_win_rates),
)


if __name__ == "__main__":
    raise SystemExit(main())
