"""Produce the combined serialization-efficiency LaTeX table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from visualizations.config import (
    CONTEXT_WINDOW_TOKENS,
    DATASET_LABELS,
    DATASET_ORDER,
    FORMAT_LABELS,
    LATEX_ROW_END,
    SERIALIZATION_EFFICIENCY_TABLE_FILENAME,
    SERIALIZATION_EFFICIENCY_TABLE_LABEL,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_LABELS,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.latex_numbers import num, percent
from visualizations.serialization_data import present_order, require_columns, sort_summary
from visualizations.uncertainty import mean_intervals, table_estimate


@dataclass(frozen=True, slots=True)
class SerializationEfficiencyAnalysis:
    """Mean token use and context-window fit summaries from one token report."""

    average_tokens: pd.DataFrame
    database_fit: pd.DataFrame


def calculate_serialization_efficiency(
    report: pd.DataFrame,
    *,
    context_window_tokens: int = CONTEXT_WINDOW_TOKENS,
) -> SerializationEfficiencyAnalysis:
    """Calculate mean token use and context-window fit by serialization format."""
    if context_window_tokens <= 0:
        raise ValueError("Context-window size must be positive")
    require_columns(
        report,
        ("dataset", "database_id", "format", "tokenizer_family", "tokenizer_label", "tokens"),
    )

    average_tokens = (
        report.groupby(["dataset", "format"], as_index=False, observed=True)
        .agg(
            mean_tokens=("tokens", "mean"),
            database_count=("database_id", "nunique"),
            tokenizer_family_count=("tokenizer_family", "nunique"),
            observation_count=("tokens", "size"),
        )
        .reset_index(drop=True)
    )
    if average_tokens.empty:
        average_tokens = average_tokens.assign(relative_to_lowest=pd.Series(dtype=float))
    else:
        minimum = average_tokens.groupby("dataset", observed=True)["mean_tokens"].transform("min")
        average_tokens["relative_to_lowest"] = average_tokens["mean_tokens"] / minimum
        average_tokens = sort_summary(average_tokens)

    intervals = mean_intervals(
        report.rename(columns={"database_id": "db_id"}), ["dataset"], {"mean_tokens": "tokens"},
        comparison="format", reference="csv", effect="ratio",
        pair_keys=("db_id", "tokenizer_family"),
    )
    average_tokens = average_tokens.merge(intervals, on=["dataset", "format"],
                                         validate="one_to_one")
    fit_data = report.copy()
    fit_data["fits_context"] = fit_data["tokens"].le(context_window_tokens)
    database_fit = (
        fit_data.groupby(
            ["dataset", "format", "tokenizer_family", "tokenizer_label"],
            as_index=False,
            observed=True,
        )
        .agg(
            fitting_databases=("fits_context", "sum"),
            total_databases=("database_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    database_fit["fitting_databases"] = database_fit["fitting_databases"].astype(int)
    database_fit["total_databases"] = database_fit["total_databases"].astype(int)
    database_fit["fit_percentage"] = (
        database_fit["fitting_databases"] / database_fit["total_databases"] * 100.0
    )
    database_fit = sort_summary(database_fit, family_column="tokenizer_family")

    return SerializationEfficiencyAnalysis(
        average_tokens=average_tokens,
        database_fit=database_fit,
    )


def render_serialization_efficiency_table(
    analysis: SerializationEfficiencyAnalysis,
    *,
    context_window_tokens: int = CONTEXT_WINDOW_TOKENS,
    coverage_validated: bool,
    caption: str | None = None,
    label: str = SERIALIZATION_EFFICIENCY_TABLE_LABEL,
) -> str:
    """Render token-use and context-window-fit metrics in one LaTeX table."""
    require_columns(
        analysis.average_tokens,
        ("dataset", "format", "mean_tokens", "relative_to_lowest"),
    )
    require_columns(
        analysis.database_fit,
        (
            "dataset",
            "format",
            "tokenizer_family",
            "fitting_databases",
            "total_databases",
            "fit_percentage",
        ),
    )
    if analysis.average_tokens.empty or analysis.database_fit.empty:
        raise ValueError("No serialization-efficiency results are available to render")
    if context_window_tokens <= 0:
        raise ValueError("Context-window size must be positive")
    if caption is None:
        caption = (
            "Serialization efficiency by dataset and format. Token use reports the mean "
            "serialized-payload token count and its within-dataset ratio to CSV. "
            f"Context fit reports payloads at or below {num(context_window_tokens)} tokens."
        )

    families = present_order(
        TOKENIZER_FAMILY_ORDER,
        analysis.database_fit["tokenizer_family"],
    )
    rows: list[str] = []
    datasets = present_order(DATASET_ORDER, analysis.average_tokens["dataset"])
    for dataset in datasets:
        average_rows = analysis.average_tokens[analysis.average_tokens["dataset"].eq(dataset)]
        fit_rows = analysis.database_fit[analysis.database_fit["dataset"].eq(dataset)]
        formats = present_order(SERIALIZATION_FORMAT_ORDER, average_rows["format"])
        for index, serialization_format in enumerate(formats):
            average_row = average_rows[average_rows["format"].eq(serialization_format)].iloc[0]
            format_fit_rows = fit_rows[fit_rows["format"].eq(serialization_format)]
            cells = [
                DATASET_LABELS.get(dataset, dataset) if index == 0 else "",
                FORMAT_LABELS.get(serialization_format, serialization_format),
                table_estimate(average_row, "mean_tokens", 0),
                table_estimate(average_row, "mean_tokens_effect", suffix=r"$\times$"),
            ]
            for family in families:
                family_rows = format_fit_rows[format_fit_rows["tokenizer_family"].eq(family)]
                if family_rows.empty:
                    cells.append("--")
                    continue
                fit_row = family_rows.iloc[0]
                cells.append(
                    f"{num(int(fit_row['fitting_databases']))}/"
                    f"{num(int(fit_row['total_databases']))} "
                    f"({percent(float(fit_row['fit_percentage']), 1)})"
                )
            rows.append("    " + " & ".join(cells) + LATEX_ROW_END)
        if dataset != datasets[-1]:
            rows.append(r"    \midrule")

    validation_note = (
        f"The complete set of {num(31)} databases, six formats, and three tokenizer "
        "families was validated."
        if coverage_validated
        else "Coverage validation was skipped."
    )
    note = (
        "Means are arithmetic averages across all databases in the named dataset and all three "
        "representative tokenizer families (60 observations per Spider format and 33 per BIRD "
        "format). Ratios are ratios of these means to CSV, not means of individual ratios. "
        "Payload-fit cells show fitting databases/total "
        "databases (percentage) for each tokenizer family. Only the serialized database payload "
        "is counted; instructions, questions, chat-template tokens, and output-token headroom "
        f"are excluded. Brackets show pointwise {percent(95)} database-bootstrap "
        "confidence intervals; "
        f"ratios use CSV as the reference. {validation_note}"
    )
    family_headers = [TOKENIZER_FAMILY_LABELS.get(family, family) for family in families]
    column_spec = "@{}llrr" + "c" * len(families) + "@{}"
    return "\n".join(
        [
            "% Generated by visualizations.serialization_efficiency_table; "
            "do not edit values manually.",
            r"\begin{table}[htbp]",
            r"  \centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            r"  \resizebox{\linewidth}{!}{%",
            f"    \\begin{{tabular}}{{{column_spec}}}",
            r"      \toprule",
            (
                r"      Dataset & Format & \multicolumn{2}{c}{Token use} & "
                 + f"\\multicolumn{{{len(families)}}}{{c}}{{Payloads at or below token limit}}"
                + LATEX_ROW_END
            ),
            (r"      \cmidrule(lr){3-4}" + f"\\cmidrule(lr){{5-{4 + len(families)}}}"),
            (
                r"       &  & Mean tokens & Ratio to CSV & "
                + " & ".join(family_headers)
                + LATEX_ROW_END
            ),
            r"      \midrule",
            *["  " + row for row in rows],
            r"      \bottomrule",
            r"    \end{tabular}%",
            r"  }",
            r"  \vspace{0.4em}",
            f"  \\parbox{{\\linewidth}}{{\\scriptsize \\textit{{Note:}} {note}}}",
            r"\end{table}",
            "",
        ]
    )


def generate_serialization_efficiency_table(
    report: pd.DataFrame,
    output_dir: Path,
    *,
    context_window_tokens: int = CONTEXT_WINDOW_TOKENS,
    coverage_validated: bool,
) -> Path:
    """Calculate and write the combined serialization-efficiency table."""
    output_path = output_dir / SERIALIZATION_EFFICIENCY_TABLE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis = calculate_serialization_efficiency(
        report,
        context_window_tokens=context_window_tokens,
    )
    output_path.write_text(
        render_serialization_efficiency_table(
            analysis,
            context_window_tokens=context_window_tokens,
            coverage_validated=coverage_validated,
        ),
        encoding="utf-8",
    )
    return output_path
