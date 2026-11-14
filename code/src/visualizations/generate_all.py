"""Generate only the tables and figures imported by the results chapter.

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
from visualizations.analysis_summaries import save_analysis_summaries
from visualizations.cohorts import (
    COHORT_COLUMNS,
    AnalysisCohorts,
    build_analysis_cohorts,
    generate_cohort_artifacts,
)
from visualizations.common_cohort_data import calculate_common_cohort_analysis
from visualizations.common_cohort_latency import generate_common_cohort_latency_figure
from visualizations.common_cohort_token_usage import generate_common_cohort_token_figure
from visualizations.config import (
    CONTEXT_WINDOW_TOKENS,
    DATASET_LABELS,
    DATASET_ORDER,
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
    calculate_direct_efficiency,
)
from visualizations.direct_operational_efficiency_table import (
    generate_direct_operational_table,
)
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
from visualizations.pipeline_retrieval_by_database_size import (
    calculate_pipeline_size_retrieval_comparison,
)
from visualizations.pipeline_retrieval_by_database_size_figure import (
    generate_pipeline_size_retrieval_figure,
)
from visualizations.pipeline_retrieval_comparison import (
    calculate_pipeline_retrieval_comparison,
    generate_pipeline_retrieval_comparison_figure,
)
from visualizations.pipeline_size_cohort_table import generate_pipeline_size_cohort_table
from visualizations.serialization_data import (
    measure_token_report,
)
from visualizations.serialization_data import (
    validate_complete_coverage as validate_serialization_coverage,
)
from visualizations.serialization_efficiency_table import (
    generate_serialization_efficiency_table,
)
from visualizations.text_to_sql_accuracy_by_model_size import (
    generate_text_to_sql_accuracy_figure,
)
from visualizations.text_to_sql_data import (
    TEXT_TO_SQL_COLUMNS,
    calculate_text_to_sql_analysis,
)
from visualizations.text_to_sql_error_correction_table import (
    generate_error_correction_table,
)
from visualizations.text_to_sql_operational_efficiency_table import (
    generate_text_to_sql_operational_table,
)
from visualizations.text_to_sql_retrieval_accuracy_table import (
    generate_text_to_sql_retrieval_table,
)
from visualizations.text_to_sql_retry_accuracy_table import generate_retry_accuracy_table
from visualizations.text_to_sql_retry_data import calculate_text_to_sql_retry_analysis
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
    cohorts: AnalysisCohorts,
    summaries_dir: Path,
) -> list[Path]:
    """Generate artifacts using the audited, explicitly selected question populations."""
    if excluded_model_markers != DEFAULT_EXCLUDED_MODEL_MARKERS:
        raise ValueError("Chapter 5 requires the confirmed ten-model roster")
    _validate_publication_cohorts(cohorts)
    answerable = cohorts.answerable_records
    context_fit = cohorts.context_fit_records
    intersection = cohorts.intersection_records
    efficiency = calculate_direct_efficiency(
        context_fit,
        excluded_model_markers=excluded_model_markers,
    )
    sql_analysis = calculate_text_to_sql_analysis(
        answerable,
        excluded_model_markers=excluded_model_markers,
    )
    retry_analysis = calculate_text_to_sql_retry_analysis(sql_analysis)
    common_cohort = calculate_common_cohort_analysis(
        context_fit,
        excluded_model_markers=excluded_model_markers,
    )
    retrieval = calculate_pipeline_retrieval_comparison(
        intersection,
        excluded_model_markers=excluded_model_markers,
    )
    size_retrieval = calculate_pipeline_size_retrieval_comparison(
        answerable,
        serialization_report,
        excluded_model_markers=excluded_model_markers,
    )
    generate_cohort_artifacts(cohorts, tables_dir=tables_dir, summaries_dir=summaries_dir)
    save_analysis_summaries(
        answerable,
        serialization_report,
        efficiency,
        sql_analysis,
        retry_analysis,
        common_cohort,
        summaries_dir,
        size_retrieval=size_retrieval,
    )
    return [
        tables_dir / "analysis_cohorts.tex",
        generate_experiment_outcomes_table(
            experiment_results,
            tables_dir,
            excluded_model_markers=excluded_model_markers,
            coverage_validated=coverage_validated,
        ),
        generate_serialization_efficiency_table(
            serialization_report,
            tables_dir,
            context_window_tokens=context_window_tokens,
            coverage_validated=coverage_validated,
        ),
        generate_tokenizer_comparison_figure(serialization_report, figures_dir),
        generate_direct_accuracy_by_format_figure(
            answerable,
            figures_dir,
            excluded_model_markers=excluded_model_markers,
        ),
        generate_direct_accuracy_by_model_size_figure(
            answerable,
            figures_dir,
            excluded_model_markers=excluded_model_markers,
        ),
        *generate_database_size_accuracy_figures(
            answerable,
            serialization_report,
            figures_dir,
            excluded_model_markers=excluded_model_markers,
        ),
        generate_direct_operational_table(efficiency, tables_dir),
        generate_text_to_sql_retrieval_table(sql_analysis, tables_dir),
        generate_text_to_sql_accuracy_figure(sql_analysis, figures_dir),
        generate_text_to_sql_operational_table(sql_analysis, tables_dir),
        generate_error_correction_table(retry_analysis, tables_dir),
        generate_retry_accuracy_table(retry_analysis, tables_dir),
        generate_pipeline_retrieval_comparison_figure(retrieval, figures_dir),
        generate_pipeline_size_cohort_table(size_retrieval, tables_dir),
        generate_pipeline_size_retrieval_figure(size_retrieval, figures_dir),
        generate_common_cohort_latency_figure(common_cohort, figures_dir),
        generate_common_cohort_token_figure(common_cohort, figures_dir),
    ]


def _validate_publication_cohorts(cohorts: AnalysisCohorts) -> None:
    problems = [
        "incomplete_questions",
        "answerable_unknown_questions",
        "context_unknown_questions",
    ]
    if cohorts.summary.empty or cohorts.summary[problems].gt(0).any(axis=None):
        raise ValueError(
            "Incomplete or unknown cohort eligibility; inspect the question-level cohort audit"
        )


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the sole visualization command's parser with project-relative defaults."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(description="Generate every results-chapter table and figure")
    parser.add_argument(
        "--results",
        type=Path,
        default=project_paths.results_dir,
        help="scored benchmark NDJSON file or directory (default: results/)",
    )
    parser.add_argument(
        "--references-dir",
        type=Path,
        default=project_paths.data_dir,
        help="directory containing stored *_references.ndjson (default: data/)",
    )
    parser.add_argument(
        "--summaries-dir",
        "--audit-dir",
        type=Path,
        default=project_paths.root / "analysis" / "summaries",
        help="CSV/JSON summaries and audit evidence (default: analysis/summaries/)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project_paths.models_dir,
        help="GGUF directory for in-memory serialization token measurements (default: models/)",
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
    """Load scored results, measure payloads in memory, and generate every artifact."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.context_window <= 0:
        parser.error("--context-window must be positive")

    excluded_markers = DEFAULT_EXCLUDED_MODEL_MARKERS
    result_columns = tuple(
        dict.fromkeys(
            (
                *OUTCOME_COLUMNS,
                *DIRECT_ACCURACY_COLUMNS,
                *DIRECT_EFFICIENCY_COLUMNS,
                *TEXT_TO_SQL_COLUMNS,
                *COHORT_COLUMNS,
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
        serialization_report = measure_token_report(models_dir=args.models_dir)
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
        print(
            f"Validated {serialization_coverage.record_count:,} serialization token counts: "
            f"{sum(serialization_coverage.database_counts.values())} databases, "
            f"{serialization_coverage.format_count} formats, "
            f"{serialization_coverage.tokenizer_family_count} tokenizer families"
        )

    try:
        cohorts = build_analysis_cohorts(
            experiment_results,
            references_dir=args.references_dir,
        )
        generate_cohort_artifacts(
            cohorts,
            tables_dir=args.tables_dir,
            summaries_dir=args.summaries_dir,
        )
        _validate_publication_cohorts(cohorts)
        direct_accuracy_count = validate_direct_accuracy_scores(cohorts.answerable_records)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(
        f"Validated retrieval-accuracy scores for {direct_accuracy_count:,} answerable direct runs"
    )

    outputs = generate_all(
        experiment_results,
        serialization_report,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        excluded_model_markers=excluded_markers,
        context_window_tokens=args.context_window,
        coverage_validated=coverage_validated,
        cohorts=cohorts,
        summaries_dir=args.summaries_dir,
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
