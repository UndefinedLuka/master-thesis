"""Generate every results-chapter table and figure through one command.

The orchestrator owns argument parsing, input loading, coverage validation, and progress output.
Artifact modules remain independently testable and each writes exactly one table or figure.

    uv run --extra visualizations thesis-visualize-all

Use ``--allow-incomplete`` only for smoke tests with partial result snapshots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.paths import ProjectPaths
from visualizations.config import (
    CONTEXT_WINDOW_TOKENS,
    DATASET_LABELS,
    DATASET_ORDER,
    EXPECTED_DATABASE_COUNTS,
)
from visualizations.data import load_scored_results
from visualizations.direct_accuracy_by_database_size import (
    generate_database_size_accuracy_figures,
)
from visualizations.direct_accuracy_by_format import (
    generate_direct_accuracy_by_format_figure,
)
from visualizations.direct_accuracy_by_model_size import (
    generate_direct_accuracy_by_model_size_figure,
)
from visualizations.direct_accuracy_data import (
    DIRECT_ACCURACY_COLUMNS,
    validate_direct_accuracy_scores,
)
from visualizations.direct_efficiency_data import (
    DIRECT_EFFICIENCY_COLUMNS,
    DirectEfficiencyAnalysis,
    calculate_direct_efficiency,
)
from visualizations.direct_latency_by_format import generate_direct_latency_figure
from visualizations.direct_operational_efficiency_table import (
    generate_direct_operational_table,
)
from visualizations.direct_token_usage import generate_direct_token_usage_figure
from visualizations.experiment_outcomes import (
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    FAILURE_OUTCOMES,
    OUTCOME_COLUMNS,
    SQL_METHOD,
    SUCCESS_OUTCOMES,
    calculate_outcome_summary,
    generate_experiment_outcomes_table,
)
from visualizations.experiment_outcomes import (
    validate_complete_coverage as validate_experiment_coverage,
)
from visualizations.serialization_average_tokens import generate_average_tokens_table
from visualizations.serialization_data import (
    load_token_report,
)
from visualizations.serialization_data import (
    validate_complete_coverage as validate_serialization_coverage,
)
from visualizations.serialization_database_fit import generate_database_fit_table
from visualizations.text_to_sql_accuracy_by_model_size import (
    generate_text_to_sql_accuracy_figure,
)
from visualizations.text_to_sql_data import (
    TEXT_TO_SQL_COLUMNS,
    TextToSqlAnalysis,
    calculate_text_to_sql_analysis,
)
from visualizations.text_to_sql_error_correction import generate_error_correction_figure
from visualizations.text_to_sql_error_correction_table import (
    generate_error_correction_table,
)
from visualizations.text_to_sql_latency import generate_text_to_sql_latency_figure
from visualizations.text_to_sql_operational_efficiency_table import (
    generate_text_to_sql_operational_table,
)
from visualizations.text_to_sql_retrieval_accuracy_table import (
    generate_text_to_sql_retrieval_table,
)
from visualizations.text_to_sql_retry_data import calculate_text_to_sql_retry_analysis
from visualizations.text_to_sql_token_usage import generate_text_to_sql_token_figure
from visualizations.tokenizer_family_comparison import generate_tokenizer_comparison_figure


def generate_all(
    experiment_results: pd.DataFrame,
    serialization_report: pd.DataFrame,
    *,
    tables_dir: Path,
    figures_dir: Path,
    excluded_model_markers: tuple[str, ...] = DEFAULT_EXCLUDED_MODEL_MARKERS,
    context_window_tokens: int = CONTEXT_WINDOW_TOKENS,
    coverage_validated: bool,
    direct_efficiency: DirectEfficiencyAnalysis | None = None,
    text_to_sql_analysis: TextToSqlAnalysis | None = None,
) -> list[Path]:
    """Generate every registered artifact from already-loaded input snapshots."""
    efficiency = direct_efficiency or calculate_direct_efficiency(
        experiment_results,
        excluded_model_markers=excluded_model_markers,
    )
    sql_analysis = text_to_sql_analysis or calculate_text_to_sql_analysis(
        experiment_results,
        excluded_model_markers=excluded_model_markers,
    )
    retry_analysis = calculate_text_to_sql_retry_analysis(sql_analysis)
    return [
        generate_experiment_outcomes_table(
            experiment_results,
            tables_dir,
            excluded_model_markers=excluded_model_markers,
            coverage_validated=coverage_validated,
        ),
        generate_average_tokens_table(
            serialization_report,
            tables_dir,
            coverage_validated=coverage_validated,
        ),
        generate_database_fit_table(
            serialization_report,
            tables_dir,
            context_window_tokens=context_window_tokens,
            coverage_validated=coverage_validated,
        ),
        generate_tokenizer_comparison_figure(serialization_report, figures_dir),
        generate_direct_accuracy_by_format_figure(
            experiment_results,
            figures_dir,
            excluded_model_markers=excluded_model_markers,
        ),
        generate_direct_accuracy_by_model_size_figure(
            experiment_results,
            figures_dir,
            excluded_model_markers=excluded_model_markers,
        ),
        *generate_database_size_accuracy_figures(
            experiment_results,
            serialization_report,
            figures_dir,
            excluded_model_markers=excluded_model_markers,
            expected_database_counts=EXPECTED_DATABASE_COUNTS if coverage_validated else None,
        ),
        generate_direct_operational_table(efficiency, tables_dir),
        generate_direct_token_usage_figure(efficiency, figures_dir),
        generate_direct_latency_figure(efficiency, figures_dir),
        generate_text_to_sql_retrieval_table(sql_analysis, tables_dir),
        generate_text_to_sql_accuracy_figure(sql_analysis, figures_dir),
        generate_text_to_sql_operational_table(sql_analysis, tables_dir),
        generate_text_to_sql_token_figure(sql_analysis, figures_dir),
        generate_text_to_sql_latency_figure(sql_analysis, figures_dir),
        generate_error_correction_table(retry_analysis, tables_dir),
        generate_error_correction_figure(retry_analysis, figures_dir),
    ]


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the sole visualization command's parser with project-relative defaults."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Generate every results-chapter table and figure"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=project_paths.results_dir,
        help="scored benchmark NDJSON file or directory (default: results/)",
    )
    parser.add_argument(
        "--serialization-report",
        type=Path,
        default=(
            project_paths.results_dir
            / "input_token_overhead"
            / "serialization_tokens_by_database.csv"
        ),
        help="serialization-token CSV produced by thesis-report-serialization-tokens",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=project_paths.root / "analysis" / "tables",
        help="LaTeX output directory (default: analysis/tables/)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=project_paths.root / "analysis" / "figures",
        help="figure output directory (default: analysis/figures/)",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=None,
        help=(
            "case-insensitive model-id fragment excluded from outcome analysis; repeat for "
            "multiple models (default: Qwen3.5-27B)"
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=CONTEXT_WINDOW_TOKENS,
        help=f"context-fit threshold in tokens (default: {CONTEXT_WINDOW_TOKENS:,})",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="skip strict experiment and serialization coverage validation (smoke tests only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load both source snapshots, validate them, and generate every artifact."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.context_window <= 0:
        parser.error("--context-window must be positive")

    excluded_markers = tuple(args.exclude_model or DEFAULT_EXCLUDED_MODEL_MARKERS)
    result_columns = tuple(
        dict.fromkeys(
            (
                *OUTCOME_COLUMNS,
                *DIRECT_ACCURACY_COLUMNS,
                *DIRECT_EFFICIENCY_COLUMNS,
                *TEXT_TO_SQL_COLUMNS,
            )
        )
    )
    experiment_results = load_scored_results(
        args.results,
        columns=result_columns,
        datasets=DATASET_ORDER,
        pipelines=("direct", SQL_METHOD),
    )
    if experiment_results.empty:
        parser.error("no scored benchmark records were found")
    try:
        serialization_report = load_token_report(args.serialization_report)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    coverage_validated = not args.allow_incomplete
    if coverage_validated:
        try:
            experiment_coverage = validate_experiment_coverage(
                experiment_results,
                excluded_model_markers=excluded_markers,
            )
            serialization_coverage = validate_serialization_coverage(serialization_report)
            direct_accuracy_count = validate_direct_accuracy_scores(
                experiment_results,
                excluded_model_markers=excluded_markers,
            )
            direct_efficiency = calculate_direct_efficiency(
                experiment_results,
                excluded_model_markers=excluded_markers,
            )
            text_to_sql_analysis = calculate_text_to_sql_analysis(
                experiment_results,
                excluded_model_markers=excluded_markers,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"Validated {experiment_coverage.run_count:,} experiment runs across "
            + ", ".join(
                f"{DATASET_LABELS.get(dataset, dataset)} "
                f"({experiment_coverage.question_counts[dataset]:,} questions)"
                for dataset in DATASET_ORDER
            )
        )
        print(f"Validated retrieval-accuracy scores for {direct_accuracy_count:,} direct runs")
        print(
            f"Validated {serialization_coverage.record_count:,} serialization token counts: "
            f"{sum(serialization_coverage.database_counts.values())} databases, "
            f"{serialization_coverage.format_count} formats, "
            f"{serialization_coverage.tokenizer_family_count} tokenizer families"
        )

    if not coverage_validated:
        direct_efficiency = calculate_direct_efficiency(
            experiment_results,
            excluded_model_markers=excluded_markers,
        )
        text_to_sql_analysis = calculate_text_to_sql_analysis(
            experiment_results,
            excluded_model_markers=excluded_markers,
        )

    outputs = generate_all(
        experiment_results,
        serialization_report,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        excluded_model_markers=excluded_markers,
        context_window_tokens=args.context_window,
        coverage_validated=coverage_validated,
        direct_efficiency=direct_efficiency,
        text_to_sql_analysis=text_to_sql_analysis,
    )
    outcome_summary = calculate_outcome_summary(
        experiment_results,
        excluded_model_markers=excluded_markers,
    )
    successful = sum(outcome_summary.outcome_total(value) for value in SUCCESS_OUTCOMES)
    failed = sum(outcome_summary.outcome_total(value) for value in FAILURE_OUTCOMES)
    print(
        f"Classified {outcome_summary.retained_run_count:,} retained runs: "
        f"{successful:,} successful, {failed:,} failed; "
        f"excluded {outcome_summary.excluded_run_count:,} model runs"
    )
    for output in outputs:
        print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
