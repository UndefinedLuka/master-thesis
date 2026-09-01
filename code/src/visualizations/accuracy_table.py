"""Generate per-dataset landscape LaTeX tables of mean accuracy by model and method.

One table is generated for BIRD and one for Spider. Rows are LLMs. Columns are the six
serialization formats plus Text-to-SQL, each split into ROUGE-L F1, BERTScore F1, and
normalized exact match. Means include every scored record, so failed outputs retain the zero
values assigned by the scoring pass. Within each dataset's method/metric column, the best model
is bold and the second-best model is underlined.

    uv run --extra visualizations python -m visualizations.accuracy_table
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from config.paths import ProjectPaths
from visualizations.data import (
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    SQL_EXECUTION_LABEL,
    load_scored_results,
)

OUTPUT_FILENAME = "mean_accuracy_by_model_and_format.tex"
METRIC_ORDER: tuple[str, ...] = (
    "rouge_l_f",
    "bertscore_f",
    "normalized_exact_match",
)
METRIC_SOURCE_COLUMNS: dict[str, str] = {
    "rouge_l_f": "scores.rouge_l_f",
    "bertscore_f": "scores.bertscore_f",
    "normalized_exact_match": "scores.normalized_exact_match",
}
METRIC_LATEX_LABELS: dict[str, str] = {
    "rouge_l_f": "R-L",
    "bertscore_f": "BERT-F",
    "normalized_exact_match": "EM",
}

DEFAULT_CAPTION = (
    "Mean retrieval-accuracy scores by model and method. Each method reports ROUGE-L F1 "
    "(R-L), BERTScore F1 (BERT-F), and normalized exact match (EM)."
)
DEFAULT_LABEL = "tab:mean-accuracy-by-model-method"


def render_dataset_accuracy_tables(
    results: pd.DataFrame,
    *,
    datasets: Sequence[str] = DATASET_ORDER,
    precision: int = 3,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> dict[str, str]:
    """Render one independently averaged and ranked accuracy table per dataset."""
    tables: dict[str, str] = {}
    for dataset in datasets:
        dataset_results = results[results["dataset"].eq(dataset)]
        summary = calculate_accuracy_means(dataset_results)
        if summary.empty:
            continue
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        tables[dataset] = render_accuracy_table(
            summary,
            precision=precision,
            caption=_caption_for_dataset(caption, dataset_label),
            label=f"{label}-{dataset}",
        )
    return tables


def output_path_for_dataset(base_path: Path, dataset: str) -> Path:
    """Append a dataset suffix to the configured base output filename."""
    return base_path.with_name(f"{base_path.stem}_{dataset}{base_path.suffix}")


def calculate_accuracy_means(results: pd.DataFrame) -> pd.DataFrame:
    """Return long-form mean scores and coverage for every model/method/metric cell."""
    required_columns = {
        "dataset",
        "pipeline",
        "format",
        "model_id",
        *METRIC_SOURCE_COLUMNS.values(),
    }
    missing = required_columns.difference(results.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Result data is missing required column(s): {names}")

    data = results.loc[:, sorted(required_columns)].copy()
    data["method"] = data["format"]
    is_sql = data["pipeline"].eq("text_to_sql") & data["format"].isna()
    data.loc[is_sql, "method"] = SQL_EXECUTION_LABEL
    data = data.dropna(subset=["dataset", "model_id", "method"])

    renamed = data.rename(
        columns={source: metric for metric, source in METRIC_SOURCE_COLUMNS.items()}
    )
    long_data = renamed.melt(
        id_vars=("dataset", "model_id", "method"),
        value_vars=METRIC_ORDER,
        var_name="metric",
        value_name="score",
    )
    long_data["score"] = pd.to_numeric(long_data["score"], errors="coerce")
    long_data = long_data.dropna(subset=["score"])
    if long_data.empty:
        return _empty_summary()

    summary = (
        long_data.groupby(["model_id", "method", "metric"], as_index=False, observed=True)
        .agg(
            mean_score=("score", "mean"),
            sample_count=("score", "size"),
            datasets=("dataset", lambda values: tuple(sorted(set(values)))),
        )
        .reset_index(drop=True)
    )
    return add_per_column_ranks(summary)


def add_per_column_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Dense-rank models from highest to lowest within each method/metric column."""
    required_columns = {"method", "metric", "mean_score"}
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Accuracy summary is missing required column(s): {names}")

    ranked = summary.copy()
    ranked["rank"] = ranked.groupby(["method", "metric"], observed=True)[
        "mean_score"
    ].rank(method="dense", ascending=False)
    return ranked


def render_accuracy_table(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str] = METHOD_ORDER,
    metrics: Sequence[str] = METRIC_ORDER,
    precision: int = 3,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> str:
    """Render the mean-score summary as a self-contained landscape table environment."""
    required_columns = {
        "model_id",
        "method",
        "metric",
        "mean_score",
        "sample_count",
        "datasets",
    }
    missing = required_columns.difference(summary.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Accuracy summary is missing required column(s): {names}")
    if summary.empty:
        raise ValueError("No mean accuracy scores are available to render")
    if precision < 0:
        raise ValueError("precision must be non-negative")

    ranked = summary if "rank" in summary.columns else add_per_column_ranks(summary)
    selected_methods = tuple(methods)
    selected_metrics = tuple(metrics)
    models = sorted(ranked["model_id"].unique(), key=model_sort_key)
    lookup = {
        (row.model_id, row.method, row.metric): row
        for row in ranked.itertuples(index=False)
    }

    column_count = len(selected_methods) * len(selected_metrics)
    group_headers = ["    "]
    sub_headers = ["    Model"]
    cmidrules: list[str] = []
    start_column = 2
    for method in selected_methods:
        method_label = METHOD_LABELS.get(method, _latex_escape(method))
        group_headers.append(
            f"\\multicolumn{{{len(selected_metrics)}}}{{c}}{{{method_label}}}"
        )
        for metric in selected_metrics:
            sub_headers.append(METRIC_LATEX_LABELS.get(metric, _latex_escape(metric)))
        end_column = start_column + len(selected_metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_column}-{end_column}}}")
        start_column = end_column + 1

    rows: list[str] = []
    for model_id in models:
        cells = [model_display_name(model_id, latex=True)]
        for method in selected_methods:
            for metric in selected_metrics:
                row = lookup.get((model_id, method, metric))
                cells.append(
                    "--"
                    if row is None
                    else _format_ranked_score(row.mean_score, row.rank, precision=precision)
                )
        rows.append("    " + " & ".join(cells) + r" \\")

    coverage_note = _coverage_note(ranked, methods=selected_methods)
    coverage_comments = _coverage_comments(ranked, models=models, methods=selected_methods)
    lines = [
        "% Generated by visualizations.accuracy_table; do not edit values manually.",
        *coverage_comments,
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"  \centering",
        f"  \\caption{{{_latex_escape(caption)}}}",
        f"  \\label{{{_latex_escape(label)}}}",
        r"  \setlength{\tabcolsep}{2.2pt}",
        r"  \renewcommand{\arraystretch}{1.15}",
        r"  \resizebox{\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{@{{}}l*{{{column_count}}}{{r}}@{{}}}}",
        r"    \toprule",
        "    " + " & ".join(group_headers) + r" \\",
        "    " + " ".join(cmidrules),
        " & ".join(sub_headers) + r" \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"  \vspace{0.5em}",
        r"  \parbox{\linewidth}{\scriptsize \textit{Note:} " + coverage_note + "}",
        r"\end{table}",
        r"\end{landscape}",
        "",
    ]
    return "\n".join(lines)


def write_latex_table(path: Path, latex: str) -> None:
    """Write a generated LaTeX table, creating its parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")


def build_parser(paths: ProjectPaths | None = None) -> argparse.ArgumentParser:
    """Build the command-line parser with project-relative default paths."""
    project_paths = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Generate one LaTeX mean-accuracy table per selected dataset"
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
        default=project_paths.root / "analysis" / "tables" / OUTPUT_FILENAME,
        help=(
            "base output path; a dataset suffix is added to each generated table "
            f"(default: analysis/tables/{OUTPUT_FILENAME})"
        ),
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
        "--precision",
        type=int,
        default=3,
        help="decimal places displayed in the table (default: 3)",
    )
    parser.add_argument("--caption", default=DEFAULT_CAPTION, help="LaTeX table caption")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="LaTeX table label")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load scored results, calculate means/ranks, and write the LaTeX table."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.precision < 0:
        parser.error("--precision must be non-negative")

    selected_datasets = _canonical_selection(DATASET_ORDER, args.dataset)
    results = load_scored_results(
        args.results,
        columns=(
            "dataset",
            "pipeline",
            "format",
            "model_id",
            *METRIC_SOURCE_COLUMNS.values(),
        ),
        datasets=selected_datasets,
        pipelines=("direct", "text_to_sql"),
        models=args.model,
    )
    tables = render_dataset_accuracy_tables(
        results,
        datasets=selected_datasets,
        precision=args.precision,
        caption=args.caption,
        label=args.label,
    )
    if not tables:
        parser.error("no scored accuracy records matched the selected filters")

    for dataset, latex in tables.items():
        path = output_path_for_dataset(args.output, dataset)
        write_latex_table(path, latex)
        dataset_summary = calculate_accuracy_means(results[results["dataset"].eq(dataset)])
        model_count = dataset_summary["model_id"].nunique()
        cell_count = len(dataset_summary)
        sample_min = int(dataset_summary["sample_count"].min())
        sample_max = int(dataset_summary["sample_count"].max())
        print(
            f"Generated {DATASET_LABELS.get(dataset, dataset)} table with {model_count} model "
            f"rows from {cell_count} metric cells (n={sample_min:,}--{sample_max:,} per cell)"
        )
        print(f"Saved LaTeX table to {path}")
    return 0


def _caption_for_dataset(caption: str, dataset_label: str) -> str:
    base = caption.strip()
    if base.endswith("."):
        base = base[:-1]
    return f"{base}. Results are reported for {dataset_label}."


def _format_ranked_score(value: float, rank: float, *, precision: int) -> str:
    formatted = f"{value:.{precision}f}"
    if rank == 1:
        return rf"\textbf{{{formatted}}}"
    if rank == 2:
        return rf"\underline{{{formatted}}}"
    return formatted


def model_display_name(model_id: str, *, latex: bool = False) -> str:
    """Return a concise model label suitable for plots or LaTeX tables."""
    separator = "~" if latex else " "
    ministral = re.search(r"Ministral-3-(\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
    if ministral:
        return f"Ministral{separator}3 {ministral.group(1)}B"
    qwen = re.search(r"Qwen3\.5-(\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
    if qwen:
        return f"Qwen{separator}3.5 {qwen.group(1)}B"
    gemma = re.search(r"gemma-4-((?:E)?\d+(?:\.\d+)?B(?:-A\d+B)?)", model_id, re.IGNORECASE)
    if gemma:
        variant = re.sub(r"b", "B", gemma.group(1), flags=re.IGNORECASE)
        return f"Gemma{separator}4 {variant}"
    fallback = model_id.removesuffix(".gguf")
    return _latex_escape(fallback) if latex else fallback


def model_sort_key(model_id: str) -> tuple[int, float, str]:
    """Sort models by family, then nominal parameter count, then identifier."""
    lowered = model_id.lower()
    if lowered.startswith("ministral"):
        family = 0
    elif lowered.startswith("qwen"):
        family = 1
    elif lowered.startswith("gemma"):
        family = 2
    else:
        family = 3
    size_match = re.search(r"(?:-|e)(\d+(?:\.\d+)?)b", lowered)
    size = float(size_match.group(1)) if size_match else float("inf")
    return family, size, lowered


def _coverage_note(summary: pd.DataFrame, *, methods: Sequence[str]) -> str:
    sample_min = int(summary["sample_count"].min())
    sample_max = int(summary["sample_count"].max())
    sample_text = f"n={sample_min:,} per cell"
    if sample_min != sample_max:
        sample_text = f"n={sample_min:,}--{sample_max:,} per cell"

    sql_rows = summary[summary["method"].eq(SQL_EXECUTION_LABEL)]
    sql_dataset_sets = {tuple(value) for value in sql_rows["datasets"]}
    coverage_warning = ""
    if SQL_EXECUTION_LABEL in methods and len(sql_dataset_sets) > 1:
        coverage_warning = (
            " Direct-format cells combine BIRD and Spider, while SQL dataset coverage varies "
            "by model because Spider SQL results are not yet available for every model."
        )

    return (
        "Values are arithmetic means over all available scored questions "
        f"({sample_text}); failed outputs retain the zero scores assigned during scoring. "
        "Within each method--metric column, the best model is \\textbf{bold} and the "
        "second-best is \\underline{underlined}. Rankings use unrounded means."
        + coverage_warning
    )


def _coverage_comments(
    summary: pd.DataFrame, *, models: Sequence[str], methods: Sequence[str]
) -> list[str]:
    cells = summary.drop_duplicates(["model_id", "method"])
    lookup = {(row.model_id, row.method): row for row in cells.itertuples(index=False)}
    comments = ["% Coverage by model and method (record count; datasets):"]
    for model_id in models:
        parts: list[str] = []
        for method in methods:
            row = lookup.get((model_id, method))
            if row is None:
                parts.append(f"{method}=missing")
            else:
                datasets = ",".join(row.datasets)
                parts.append(f"{method}=n{row.sample_count}[{datasets}]")
        comments.append(f"% {model_id}: " + "; ".join(parts))
    return comments


def _canonical_selection(order: Sequence[str], selected: Sequence[str] | None) -> tuple[str, ...]:
    if selected is None:
        return tuple(order)
    selected_set = set(selected)
    return tuple(value for value in order if value in selected_set)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "model_id",
            "method",
            "metric",
            "mean_score",
            "sample_count",
            "datasets",
            "rank",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
