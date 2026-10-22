"""Render answerable-cohort coverage for the head-to-head database-size analysis."""

from __future__ import annotations

from pathlib import Path

from visualizations.config import LATEX_ROW_END
from visualizations.latex_numbers import num
from visualizations.pipeline_retrieval_by_database_size import (
    SIZE_TIER_LABELS,
    SIZE_TIER_ORDER,
    PipelineSizeRetrievalAnalysis,
)

PIPELINE_SIZE_COHORT_TABLE_FILENAME = "pipeline_size_cohorts.tex"
SIZE_TIER_RANGES = {
    "small": rf"$\leq$ {num(1500)}",
    "medium": rf"$>\!$ {num(1500)} and $\leq$ {num(10000)}",
    "large": rf"$>\!$ {num(10000)}",
}


def render_pipeline_size_cohort_table(analysis: PipelineSizeRetrievalAnalysis) -> str:
    """Report one pooled population without dataset-specific rows."""
    coverage = analysis.coverage.set_index("size_tier")
    rows = []
    for tier in SIZE_TIER_ORDER:
        counts = coverage.loc[tier]
        rows.append(
            f"    {SIZE_TIER_LABELS[tier]} & {SIZE_TIER_RANGES[tier]} & "
            f"{num(int(counts['database_count']))} & {num(int(counts['question_count']))}"
            + LATEX_ROW_END
        )
    return "\n".join([
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \caption{Pooled answerable-cohort coverage by database-size tier. Size is the mean payload",
        r"    token count across six formats and three tokenizer families. Each question is",
        r"    evaluated across the ten retained models for every retrieval configuration.}",
        r"  \label{tab:pipeline-size-cohorts}",
        r"  \small",
        r"  \begin{tabular}{@{}llrr@{}}",
        r"    \toprule",
        r"    Tier & Mean payload tokens & Databases & Questions" + LATEX_ROW_END,
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])


def generate_pipeline_size_cohort_table(
    analysis: PipelineSizeRetrievalAnalysis, tables_dir: Path,
) -> Path:
    """Write one coverage table for the supplied answerable-cohort analysis."""
    path = tables_dir / PIPELINE_SIZE_COHORT_TABLE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_pipeline_size_cohort_table(analysis), encoding="utf-8")
    return path
