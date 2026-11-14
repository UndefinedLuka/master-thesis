"""Calculate the retry-accuracy table without inference, rescoring, or plotting.

Run ``python -m visualizations.text_to_sql_retry_accuracy`` from ``code/``.
The output uses the answerable cohort and ten-model roster of the existing
error-correction analysis, not the head-to-head intersection cohort. Accuracy
means include only successfully recovered runs; failures are counted separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.paths import ProjectPaths
from visualizations.cohorts import COHORT_COLUMNS, build_analysis_cohorts
from visualizations.config import DATASET_LABELS, DATASET_ORDER
from visualizations.data import load_scored_results
from visualizations.text_to_sql_data import TEXT_TO_SQL_COLUMNS, calculate_text_to_sql_analysis
from visualizations.text_to_sql_retry_accuracy_table import generate_retry_accuracy_table
from visualizations.text_to_sql_retry_data import calculate_text_to_sql_retry_analysis


def main(argv: list[str] | None = None) -> int:
    """Audit cohort eligibility and save unrounded cumulative retry-accuracy estimates."""
    paths = ProjectPaths.discover()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=paths.results_dir)
    parser.add_argument("--references-dir", type=Path, default=paths.data_dir)
    parser.add_argument(
        "--output", type=Path,
        default=paths.root / "analysis" / "summaries" / "sql_retry_accuracy.csv",
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=paths.root / "analysis" / "tables",
    )
    args = parser.parse_args(argv)
    try:
        results = load_scored_results(
            args.results,
            columns=tuple(dict.fromkeys((*COHORT_COLUMNS, *TEXT_TO_SQL_COLUMNS))),
            datasets=DATASET_ORDER,
            pipelines=("direct", "text_to_sql"),
        )
        cohorts = build_analysis_cohorts(results, references_dir=args.references_dir)
        problems = [
            "incomplete_questions", "answerable_unknown_questions", "context_unknown_questions",
        ]
        if cohorts.summary.empty or cohorts.summary[problems].gt(0).any(axis=None):
            raise ValueError(
                "Incomplete or unknown cohort eligibility; cannot publish retry accuracy"
            )
        sql = calculate_text_to_sql_analysis(cohorts.answerable_records)
        retries = calculate_text_to_sql_retry_analysis(sql)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    summary = retries.accuracy_summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    table_path = generate_retry_accuracy_table(retries, args.tables_dir)
    for dataset in DATASET_ORDER:
        selected = summary.loc[summary["dataset"].eq(dataset)]
        if selected.empty:
            continue
        counts = selected.iloc[0]
        print(
            f"{DATASET_LABELS[dataset]}: {int(counts['run_count']):,} question-model runs "
            f"entering correction ({int(counts['question_count']):,} unique questions), "
            f"out of {int(counts['eligible_run_count']):,} answerable runs"
        )
        print(selected[[
            "retry_label", "newly_executed_run_count", "cumulative_executed_run_count",
            "unrecovered_run_count", "normalized_exact_match", "rouge_l_f", "bertscore_f",
        ]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved {args.output}")
    print(f"Saved {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
