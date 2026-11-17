"""Calculate answerable database-size associations for all metrics; plot NEM only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from visualizations.config import (
    DATASET_COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    DIRECT_ACCURACY_METRIC_ORDER,
    DIRECT_DATABASE_SIZE_FIGURE_FILENAMES,
    FORMAT_LABELS,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.direct_accuracy_data import prepare_direct_accuracy
from visualizations.plot_numbers import number, number_formatter, save_pgf_figure, tight_layout
from visualizations.serialization_data import require_columns
from visualizations.uncertainty import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class DatabaseSizeAnalysis:
    """Format-specific database accuracy, sizes, and dataset correlations."""

    database_scores: pd.DataFrame
    correlations: pd.DataFrame


def calculate_database_size_accuracy(
    results: pd.DataFrame,
    serialization_report: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> DatabaseSizeAnalysis:
    """Aggregate accuracy and effective size for each database and serialization format.

    A database-format size is the mean token count across the three representative tokenizer
    families. Accuracy is the end-to-end mean across every retained model and question for the
    same database and format in the supplied answerable cohort; explicit zero scores from
    context-window failures remain included. Associations use database-level means, with
    the existing database-resampling intervals, rather than treating runs as independent.
    """
    require_columns(
        serialization_report,
        ("dataset", "database_id", "format", "tokenizer_family", "row_count",
         "table_count", "tokens"),
    )
    accuracy = prepare_direct_accuracy(
        results,
        excluded_model_markers=excluded_model_markers,
    )
    if accuracy.empty:
        raise ValueError("No direct-context accuracy records are available")

    database_accuracy = (
        accuracy.groupby(
            ["dataset", "db_id", "format", "metric"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_score=("score", "mean"),
            observation_count=("score", "size"),
            question_count=("example_id", "nunique"),
            model_count=("model_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    sizes = (
        serialization_report.groupby(
            ["dataset", "database_id", "format"],
            as_index=False,
            observed=True,
        )
        .agg(
            row_count=("row_count", "first"),
            row_count_variants=("row_count", "nunique"),
            table_count=("table_count", "first"),
            table_count_variants=("table_count", "nunique"),
            mean_serialized_tokens=("tokens", "mean"),
            tokenizer_family_count=("tokenizer_family", "nunique"),
        )
        .reset_index(drop=True)
    )
    inconsistent = sizes[
        sizes["row_count_variants"].ne(1) | sizes["table_count_variants"].ne(1)
    ]
    if not inconsistent.empty:
        row = inconsistent.iloc[0]
        raise ValueError(
            f"Database size metadata differs across token records for "
            f"{row['dataset']}/{row['database_id']}/{row['format']}"
        )
    sizes = sizes.drop(columns=["row_count_variants", "table_count_variants"])

    merged = database_accuracy.merge(
        sizes,
        left_on=["dataset", "db_id", "format"],
        right_on=["dataset", "database_id", "format"],
        how="left",
        validate="many_to_one",
    )
    missing_size = merged["mean_serialized_tokens"].isna()
    if missing_size.any():
        missing = merged.loc[missing_size, ["dataset", "db_id", "format"]].iloc[0]
        raise ValueError(f"Missing serialization size for database-format: {missing.to_dict()}")
    if (not np.isfinite(merged["mean_serialized_tokens"]).all()
            or merged["mean_serialized_tokens"].le(0).any()):
        raise ValueError("Database effective size must be positive")

    _validate_database_format_coverage(merged, accuracy)

    merged["log10_serialized_tokens"] = merged["mean_serialized_tokens"].map(math.log10)
    merged = merged.sort_values(
        ["metric", "format", "dataset", "mean_serialized_tokens", "db_id"]
    ).reset_index(drop=True)

    correlation_rows: list[dict[str, object]] = []
    for (dataset, serialization_format, metric), group in merged.groupby(
        ["dataset", "format", "metric"], observed=True
    ):
        lower, upper, _, _ = _bootstrap_size_association(group)
        correlation_rows.append(
            {
                "dataset": dataset,
                "format": serialization_format,
                "metric": metric,
                "pearson_log_size": _pearson_correlation(
                    group["log10_serialized_tokens"],
                    group["mean_score"],
                ),
                "database_count": len(group),
                "pearson_log_size_low": lower,
                "pearson_log_size_high": upper,
            }
        )
    correlations = pd.DataFrame(correlation_rows).sort_values(
        ["metric", "format", "dataset"]
    ).reset_index(drop=True)
    return DatabaseSizeAnalysis(database_scores=merged, correlations=correlations)


def plot_database_size_accuracy(
    analysis: DatabaseSizeAnalysis,
) -> tuple[Figure, list[Axes]]:
    """Plot NEM in three rows of two format panels, each showing both datasets."""
    metric = "normalized_exact_match"
    data = analysis.database_scores
    required = {
        "dataset",
        "db_id",
        "format",
        "metric",
        "mean_score",
        "mean_serialized_tokens",
        "question_count",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Database-size summary is missing: {', '.join(sorted(missing))}")
    panel_data = data[data["metric"].eq(metric)]
    if panel_data.empty:
        raise ValueError(f"No database-size accuracy values are available for metric: {metric}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    formats = [
        value for value in SERIALIZATION_FORMAT_ORDER if value in set(panel_data["format"])
    ]
    correlation_lookup = {
        (row.dataset, row.format): float(row.pearson_log_size)
        for row in analysis.correlations[analysis.correlations["metric"].eq(metric)].itertuples(
            index=False
        )
    }
    figure, axes_array = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(11.2, 15.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_array.flat)
    y_limits = _score_limits(panel_data["mean_score"])

    for axis, serialization_format in zip(axes[: len(formats)], formats, strict=True):
        format_rows = panel_data[panel_data["format"].eq(serialization_format)]
        for dataset in DATASET_ORDER:
            dataset_rows = format_rows[format_rows["dataset"].eq(dataset)].sort_values(
                "mean_serialized_tokens"
            )
            if dataset_rows.empty:
                continue
            marker_sizes = [
                34.0 + math.sqrt(float(count)) * 3.2
                for count in dataset_rows["question_count"]
            ]
            axis.scatter(
                dataset_rows["mean_serialized_tokens"],
                dataset_rows["mean_score"],
                s=marker_sizes,
                color=DATASET_COLORS[dataset],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.82,
                zorder=3,
            )
            trend_x, trend_y = _linear_trend(
                dataset_rows["mean_serialized_tokens"],
                dataset_rows["mean_score"],
            )
            axis.plot(
                trend_x,
                trend_y,
                color=DATASET_COLORS[dataset],
                linewidth=2.0,
                linestyle="--",
                alpha=0.9,
            )
        axis.set_xscale("log")
        axis.set_ylim(*y_limits)
        axis.yaxis.set_major_formatter(number_formatter())
        axis.grid(axis="both", color="#D0D0D0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(
            FORMAT_LABELS.get(serialization_format, serialization_format),
            fontweight="bold",
        )
        axis.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
        axis.text(
            0.97,
            0.96,
            "Pearson $r$ vs. log size",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#333333",
        )
        for correlation_index, dataset in enumerate(DATASET_ORDER):
            correlation = correlation_lookup.get((dataset, serialization_format), math.nan)
            if math.isnan(correlation):
                continue
            selected = format_rows[format_rows["dataset"].eq(dataset)]
            lower, upper, grid, band = _bootstrap_size_association(selected)
            interval = "Interval unavailable"
            if np.isfinite([lower, upper]).all():
                interval = f"[{number(lower, 2)}; {number(upper, 2)}]"
            if band is not None:
                axis.fill_between(10 ** grid, *band, color=DATASET_COLORS[dataset], alpha=0.10)
            axis.text(
                0.97,
                0.89 - correlation_index * 0.065,
                f"{DATASET_LABELS[dataset]}: {number(correlation, 2)} {interval}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color=DATASET_COLORS[dataset],
                fontweight="bold",
            )

    for axis in axes[len(formats) :]:
        axis.set_visible(False)
    figure.supxlabel(
        "Serialised database size (tokens, log scale)",
    )
    for axis in axes[::2]:
        if axis.get_visible():
            axis.set_ylabel("Mean NEM")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=DATASET_COLORS[dataset],
            linestyle="--",
            linewidth=2,
            markersize=8,
            label=DATASET_LABELS[dataset],
        )
        for dataset in DATASET_ORDER
        if dataset in set(panel_data["dataset"])
    ]
    figure.legend(
        handles=handles,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=False,
    )
    tight_layout(figure, rect=(0.03, 0, 1.0, 0.96))
    return figure, axes


def generate_database_size_accuracy_figures(
    results: pd.DataFrame,
    serialization_report: pd.DataFrame,
    output_dir: Path,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> list[Path]:
    """Calculate all three associations and save only the six-format NEM figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = calculate_database_size_accuracy(
        results,
        serialization_report,
        excluded_model_markers=excluded_model_markers,
    )
    output_path = output_dir / DIRECT_DATABASE_SIZE_FIGURE_FILENAMES["normalized_exact_match"]
    figure, _ = plot_database_size_accuracy(analysis)

    import matplotlib.pyplot as plt

    save_pgf_figure(figure, output_path)
    plt.close(figure)
    return [output_path]


def _validate_database_format_coverage(
    scores: pd.DataFrame,
    answerable_accuracy: pd.DataFrame,
) -> None:
    """Require the actual answerable database identities in every format and metric."""
    for dataset, cohort in answerable_accuracy.groupby("dataset", observed=True):
        expected = set(cohort["db_id"])
        dataset_rows = scores[scores["dataset"].eq(dataset)]
        for serialization_format in SERIALIZATION_FORMAT_ORDER:
            for metric in DIRECT_ACCURACY_METRIC_ORDER:
                actual = set(dataset_rows.loc[
                    dataset_rows["format"].eq(serialization_format)
                    & dataset_rows["metric"].eq(metric), "db_id"
                ])
                if actual != expected:
                    raise ValueError(
                        f"Answerable accuracy database coverage differs for "
                        f"{dataset}/{serialization_format}/{metric}: "
                        f"missing {sorted(expected - actual)}, "
                        f"unexpected {sorted(actual - expected)}"
                    )


def _bootstrap_size_association(
    group: pd.DataFrame,
) -> tuple[float, float, np.ndarray, np.ndarray | None]:
    """Preserve the existing paired database bootstrap and degeneracy threshold.

    Resample whole database points 10,000 times. Report pointwise percentile intervals
    only with at least three databases and at least 95% non-degenerate draws.
    """
    x = group["log10_serialized_tokens"].to_numpy()
    y = group["mean_score"].to_numpy()
    grid = np.linspace(x.min(), x.max(), 100)
    lower = upper = math.nan
    band = None
    if len(x) < 3:
        return lower, upper, grid, band
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_DRAWS, len(x)))
    bx, by = x[indices], y[indices]
    dx, dy = bx - bx.mean(axis=1)[:, None], by - by.mean(axis=1)[:, None]
    xx, yy = (dx * dx).sum(axis=1), (dy * dy).sum(axis=1)
    xy = (dx * dy).sum(axis=1)
    valid = (xx > 0) & (yy > 0)
    if valid.mean() >= 0.95:
        draws = xy[valid] / np.sqrt(xx[valid] * yy[valid])
        lower, upper = (float(value) for value in np.quantile(draws, [0.025, 0.975]))
    valid_trend = xx > 0
    if valid_trend.mean() >= 0.95:
        slopes = xy[valid_trend] / xx[valid_trend]
        predictions = (
            by[valid_trend].mean(axis=1)[:, None]
            + slopes[:, None] * (grid - bx[valid_trend].mean(axis=1)[:, None])
        )
        band = np.quantile(predictions, [0.025, 0.975], axis=0)
    return lower, upper, grid, band


def _pearson_correlation(x_values: pd.Series, y_values: pd.Series) -> float:
    if len(x_values) < 2 or x_values.nunique() < 2 or y_values.nunique() < 2:
        return math.nan
    return float(x_values.corr(y_values))


def _linear_trend(
    sizes: pd.Series,
    scores: pd.Series,
) -> tuple[list[float], list[float]]:
    x_values = [math.log10(float(value)) for value in sizes]
    y_values = [float(value) for value in scores]
    if len(x_values) < 2:
        return [10**x_values[0]], [y_values[0]]
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    denominator = sum((value - mean_x) ** 2 for value in x_values)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values, strict=True))
        / denominator
        if denominator > 0
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    bounds = [min(x_values), max(x_values)]
    return [10**value for value in bounds], [intercept + slope * value for value in bounds]


def _score_limits(values: pd.Series) -> tuple[float, float]:
    numeric = [float(value) for value in values if not pd.isna(value)]
    lower = min(0.0, min(numeric, default=0.0))
    upper = max(numeric, default=1.0)
    margin = max(0.03, (upper - lower) * 0.12)
    return lower - (0.02 if lower < 0 else 0.0), upper + margin


def _compact_tick(value: float, _position: float) -> str:
    if value >= 1_000_000:
        return number(value / 1_000_000) + r"\,M"
    if value >= 1_000:
        return number(value / 1_000) + r"\,k"
    return number(value)
