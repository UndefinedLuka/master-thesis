"""Paired retrieval accuracy on explicitly selected context-fit ∩ answerable records.

Cohort selection belongs to the caller. Failed outputs remain observations, and SQL
is stored once per question/model rather than repeated for each direct format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from visualizations.config import (
    COMMON_COHORT_METHOD_ORDER,
    COMMON_COHORT_SQL_METHOD,
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    DIRECT_ACCURACY_METRIC_ORDER,
    FORMAT_COLORS,
    FORMAT_LABELS,
    METRIC_TITLES,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.experiment_outcomes import model_is_excluded
from visualizations.plot_numbers import number, number_formatter, save_pgf_figure, tight_layout
from visualizations.uncertainty import interval_marks, mean_intervals

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PIPELINE_RETRIEVAL_FIGURE_FILENAME = "pipeline_retrieval_comparison.pdf"
PAIR_KEYS = ("dataset", "db_id", "example_id", "model_id")
SCORE_COLUMNS = {metric: f"scores.{metric}" for metric in DIRECT_ACCURACY_METRIC_ORDER}


@dataclass(frozen=True, slots=True)
class PipelineRetrievalComparison:
    """Unique run records and long-form absolute means/paired direct-minus-SQL effects."""

    records: pd.DataFrame
    summary: pd.DataFrame


def calculate_pipeline_retrieval_comparison(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> PipelineRetrievalComparison:
    """Compare preselected context-fit ∩ answerable runs using database bootstrap CIs.

    The caller supplies the context-fit ∩ answerable questions, using the cohort manifest
    and reference SQL rather than selecting successful outputs.
    All six direct formats and SQL must have identical dataset/database/question/model
    identities. No output-success or context-fit selection is performed here. Means
    give each run equal weight; pointwise 95% intervals use the shared 10,000-draw
    database bootstrap, with the same sampled databases for every method. Intervals
    are omitted for groups with fewer than three databases, including the two-database
    BIRD context-fit subset; point estimates and paired effects are retained.
    """
    required = {*PAIR_KEYS, "pipeline", "format", "error", "execution_success",
                *SCORE_COLUMNS.values()}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Pipeline retrieval data is missing: {', '.join(sorted(missing))}")
    data = results.loc[~results["model_id"].map(
        lambda model: model_is_excluded(model, excluded_model_markers)
    )].copy()
    if data.empty:
        raise ValueError("No retained answerable-cohort records are available")
    if data[list(PAIR_KEYS)].isna().any(axis=None):
        raise ValueError("Missing pipeline retrieval pairing identity")
    if not data["dataset"].isin(DATASET_ORDER).all():
        raise ValueError("Unsupported dataset in pipeline retrieval comparison")
    sql = data["pipeline"].eq(COMMON_COHORT_SQL_METHOD)
    direct = data["pipeline"].eq("direct")
    if not (sql | direct).all():
        raise ValueError("Unsupported pipeline in retrieval comparison")
    if not data.loc[direct, "format"].isin(SERIALIZATION_FORMAT_ORDER).all():
        raise ValueError("Unsupported direct serialisation format")
    if data.loc[sql, "format"].notna().any():
        raise ValueError("Text-to-SQL records must have no serialisation format")
    if not data.loc[sql, "execution_success"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("SQL execution_success must be an explicit boolean")
    data["method"] = data["format"].where(direct, COMMON_COHORT_SQL_METHOD)
    if data.duplicated([*PAIR_KEYS, "method"]).any():
        raise ValueError("Duplicate pipeline retrieval run identity")
    database_counts = data.groupby(["dataset", "example_id"], observed=True)["db_id"].nunique()
    if database_counts.ne(1).any():
        raise ValueError("Question identities map to inconsistent databases")
    for dataset, group in data.groupby("dataset", observed=True):
        if set(group["method"]) != set(COMMON_COHORT_METHOD_ORDER):
            raise ValueError(f"{dataset}: all six direct formats and SQL are required")

    # Match metrics.scoring._output_failed, including errors with empty-string messages.
    failed = data["error"].notna() | (sql & data["execution_success"].eq(False))
    for metric, column in SCORE_COLUMNS.items():
        values = pd.to_numeric(data[column], errors="coerce")
        lower = -1.0 if metric == "bertscore_f" else 0.0
        # The scorer stores raw, unrescaled BERTScore without clipping. Float32
        # cosine/reduction roundoff can exceed 1: the campaign maximum is
        # 1.0000011870480279. Allow small numerical error at both bounds while
        # preserving the exact stored scores in records, means and paired effects.
        tolerance = 1e-5 if metric == "bertscore_f" else 0.0
        valid = np.isfinite(values) & values.between(lower - tolerance, 1 + tolerance)
        if metric == "normalized_exact_match":
            valid &= values.isin([0, 1])
        if not valid.all():
            raise ValueError(f"Invalid {metric} score(s) in pipeline retrieval data")
        if values.loc[failed].ne(0).any():
            raise ValueError(f"Failed outputs must have zero {metric} scores")
        data[column] = values

    long = data.melt(
        id_vars=[*PAIR_KEYS, "method"], value_vars=list(SCORE_COLUMNS.values()),
        var_name="metric", value_name="score",
    )
    long["metric"] = long["metric"].map(
        {source: metric for metric, source in SCORE_COLUMNS.items()}
    )
    summary = long.groupby(["dataset", "method", "metric"], as_index=False, observed=True).agg(
        mean_score=("score", "mean"), observation_count=("score", "size"),
        question_count=("example_id", "nunique"), model_count=("model_id", "nunique"),
        database_count=("db_id", "nunique"),
    )
    intervals = mean_intervals(
        long, ["dataset", "metric"], {"mean_score": "score"}, comparison="method",
        reference=COMMON_COHORT_SQL_METHOD, pair_keys=PAIR_KEYS,
        min_databases=3,
    )
    summary = summary.merge(intervals, on=["dataset", "metric", "method"], validate="one_to_one")
    order = {"dataset": DATASET_ORDER, "metric": DIRECT_ACCURACY_METRIC_ORDER,
             "method": COMMON_COHORT_METHOD_ORDER}
    summary = summary.sort_values(
        list(order), key=lambda values: values.map({v: i for i, v in enumerate(order[values.name])})
    ).reset_index(drop=True)
    return PipelineRetrievalComparison(records=data.reset_index(drop=True), summary=summary)


def plot_pipeline_retrieval_comparison(analysis: PipelineRetrievalComparison) -> Figure:
    """Compare SQL and direct formats with metric rows and dataset columns."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = analysis.summary
    if summary.empty:
        raise ValueError("No pipeline retrieval comparisons are available to plot")
    datasets = [value for value in DATASET_ORDER if value in set(summary["dataset"])]
    methods = COMMON_COHORT_METHOD_ORDER
    positions = np.arange(len(methods), dtype=float)
    positions[1:] += 0.4
    labels = ["Text-to-SQL", *(FORMAT_LABELS[fmt] for fmt in SERIALIZATION_FORMAT_ORDER)]
    colors = {COMMON_COHORT_SQL_METHOD: "#222222", **FORMAT_COLORS}
    figure, axes = plt.subplots(
        len(DIRECT_ACCURACY_METRIC_ORDER), len(datasets),
        figsize=(3.0 * len(datasets), 2.9 * len(DIRECT_ACCURACY_METRIC_ORDER)),
        squeeze=False, sharex="row", sharey=True,
    )
    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(DIRECT_ACCURACY_METRIC_ORDER):
            axis = axes[row, col]
            panel = summary.loc[summary["dataset"].eq(dataset) & summary["metric"].eq(metric)]
            panel = panel.set_index("method").loc[list(methods)]
            for position, method in zip(positions, methods, strict=True):
                color = colors[method]
                interval_marks(axis, [position], panel.loc[[method]], "mean_score",
                               horizontal=True, color=color)
                mean = panel.loc[method, "mean_score"]
                axis.plot(mean, position, "o", color=color, markersize=4, zorder=6)
                axis.annotate(number(mean, 3), (mean, position), xytext=(0, 5),
                              textcoords="offset points", ha="center", va="bottom",
                              color=color)
            axis.axhline(0.7, color="#dddddd", linewidth=0.6)
            axis.set_yticks(positions, labels)
            axis.set_ylim(positions[-1] + 0.5, -0.9)
            # Use the same score scale in every panel unless negative BERTScore
            # means/intervals require extending that metric's shared axis.
            metric_values = summary.loc[summary["metric"].eq(metric),
                                        ["mean_score", "mean_score_low", "mean_score_high"]]
            lower = min(0.0, float(metric_values.min().min()))
            upper = max(1.0, float(metric_values.max().max()))
            axis.set_xlim(lower - 0.025, upper + 0.025)
            axis.set_xticks(np.linspace(0, 1, 5))
            if row == 0:
                axis.set_title(DATASET_LABELS[dataset], pad=12)
            if col == 0:
                axis.set_ylabel(METRIC_TITLES[metric], labelpad=12)
                axis.get_yticklabels()[0].set_color(colors[COMMON_COHORT_SQL_METHOD])
            axis.xaxis.set_major_formatter(number_formatter())
            axis.tick_params(axis="y", length=0)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.spines["bottom"].set_color("#aaaaaa")
            axis.set_axisbelow(True)
            axis.grid(axis="x", color="#e8e8e8", linewidth=0.6)
    figure.supxlabel("Mean score")
    tight_layout(figure, h_pad=1.8, w_pad=1.5)
    return figure


def generate_pipeline_retrieval_comparison_figure(
    analysis: PipelineRetrievalComparison, figures_dir: Path,
) -> Path:
    """Write the absolute-score accuracy PDF from an already validated analysis."""
    import matplotlib.pyplot as plt

    path = figures_dir / PIPELINE_RETRIEVAL_FIGURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_pipeline_retrieval_comparison(analysis)
    try:
        save_pgf_figure(figure, path)
    finally:
        plt.close(figure)
    return path
