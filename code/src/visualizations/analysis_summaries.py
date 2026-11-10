"""Export accuracy, efficiency, retry, and pipeline-comparison summaries to CSV."""

from pathlib import Path

from visualizations.direct_accuracy_by_database_size import calculate_database_size_accuracy
from visualizations.direct_accuracy_data import (
    calculate_format_accuracy,
    calculate_model_size_accuracy,
)


def save_analysis_summaries(
    answerable,
    serialization_report,
    efficiency,
    sql,
    retries,
    common,
    output_dir: Path,
    *,
    size_retrieval,
) -> None:
    """Save run-weighted means, database intervals, and paired effects without rounding."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = calculate_database_size_accuracy(answerable, serialization_report)
    summaries = {
        "direct_accuracy_by_format": calculate_format_accuracy(answerable),
        "direct_accuracy_by_model": calculate_model_size_accuracy(answerable),
        "direct_database_scores": sizes.database_scores,
        "direct_size_correlations": sizes.correlations,
        "direct_efficiency": efficiency.summary,
        "sql_accuracy_and_efficiency": sql.dataset_summary,
        "sql_model_accuracy_and_efficiency": sql.model_summary,
        "sql_retry_outcomes": retries.outcome_summary,
        "sql_retry_stages": retries.retry_stage_summary,
        "sql_retry_gains": retries.gain_summary,
        "pipeline_efficiency": common.summary,
        "pipeline_database_size_tiers": size_retrieval.database_sizes,
        "pipeline_size_cohorts": size_retrieval.coverage,
        "pipeline_retrieval_by_database_size": size_retrieval.summary,
    }
    for name, summary in summaries.items():
        summary.to_csv(output_dir / f"{name}.csv", index=False)
    pair_summary = (
        common.paired_comparisons.groupby(["dataset", "format"])
        .agg(
            sql_faster_fraction=("sql_faster", "mean"),
            pair_count=("example_id", "size"),
        )
        .reset_index()
    )
    pair_summary.to_csv(output_dir / "pipeline_latency_pairs.csv", index=False)
    failures = (
        common.records[common.records["pipeline"].eq("text_to_sql")]
        .groupby("dataset")
        .agg(
            run_count=("example_id", "size"),
            execution_success_count=("execution_success", "sum"),
        )
    )
    failures.to_csv(output_dir / "context_fit_sql_outcomes.csv")
