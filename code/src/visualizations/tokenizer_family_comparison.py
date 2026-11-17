"""Produce the paired tokenizer-family comparison figure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from visualizations.config import (
    DATASET_LABELS,
    DATASET_ORDER,
    FORMAT_LABELS,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_COMPARISON_FIGURE_FILENAME,
    TOKENIZER_FAMILY_COLORS,
    TOKENIZER_FAMILY_LABELS,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.plot_numbers import number, number_formatter, save_pgf_figure, tight_layout
from visualizations.serialization_data import present_order, require_columns, sort_summary
from visualizations.uncertainty import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    bounds,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def calculate_tokenizer_overhead(report: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired tokenizer overhead relative to the best family for each payload."""
    require_columns(
        report,
        ("dataset", "database_id", "format", "tokenizer_family", "tokenizer_label", "tokens"),
    )
    data = report.copy()
    paired_keys = ["dataset", "database_id", "format"]
    best_tokens = data.groupby(paired_keys, observed=True)["tokens"].transform("min")
    data["overhead_pct"] = (data["tokens"] / best_tokens - 1.0) * 100.0
    summary = (
        data.groupby(
            ["dataset", "format", "tokenizer_family", "tokenizer_label"],
            as_index=False,
            observed=True,
        )
        .agg(
            median_overhead_pct=("overhead_pct", "median"),
            q1_overhead_pct=("overhead_pct", lambda values: values.quantile(0.25)),
            q3_overhead_pct=("overhead_pct", lambda values: values.quantile(0.75)),
            mean_overhead_pct=("overhead_pct", "mean"),
            database_count=("database_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    intervals = []
    for (dataset, fmt), group in data.groupby(["dataset", "format"], observed=True):
        tokens = group.pivot(index="database_id", columns="tokenizer_family", values="tokens")
        if tokens.isna().any(axis=None):
            raise ValueError("Tokenizer comparisons require identical databases")
        reference = "qwen-3.5"
        if reference not in tokens:
            raise ValueError("Tokenizer comparison requires Qwen 3.5 as reference")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(0, len(tokens), size=(BOOTSTRAP_DRAWS, len(tokens)))
        for family in tokens:
            overhead = ((tokens[family] / tokens.min(axis=1) - 1) * 100).to_numpy()
            ratio = (tokens[family] / tokens[reference]).to_numpy()
            low, high = bounds(np.median(overhead[indices], axis=1))
            ratio_low, ratio_high = bounds(np.median(ratio[indices], axis=1))
            if len(tokens) < 2:
                low = high = ratio_low = ratio_high = float("nan")
            intervals.append({
                "dataset": dataset, "format": fmt, "tokenizer_family": family,
                "median_overhead_pct_low": low, "median_overhead_pct_high": high,
                "tokens_effect": float(np.median(ratio)),
                "tokens_effect_low": ratio_low, "tokens_effect_high": ratio_high,
            })
    summary = summary.merge(pd.DataFrame(intervals),
                            on=["dataset", "format", "tokenizer_family"], validate="one_to_one")
    return sort_summary(summary, family_column="tokenizer_family")


def plot_tokenizer_comparison(summary: pd.DataFrame) -> Figure:
    """Plot paired tokenizer overhead medians and confidence intervals."""
    require_columns(
        summary,
        (
            "dataset",
            "format",
            "tokenizer_family",
            "median_overhead_pct",
            "q1_overhead_pct",
            "q3_overhead_pct",
            "database_count",
        ),
    )
    if summary.empty:
        raise ValueError("No tokenizer-overhead values are available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = present_order(DATASET_ORDER, summary["dataset"])
    formats = present_order(SERIALIZATION_FORMAT_ORDER, summary["format"])
    families = present_order(TOKENIZER_FAMILY_ORDER, summary["tokenizer_family"])
    figure, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(10.0, 4.1 * len(datasets)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    if len(families) > 1:
        offset_step = 0.44 / (len(families) - 1)
        offsets = [-0.22 + index * offset_step for index in range(len(families))]
    else:
        offsets = [0.0]
    x_positions = list(range(len(formats)))

    for dataset_index, dataset in enumerate(datasets):
        axis = axes[dataset_index, 0]
        dataset_rows = summary[summary["dataset"].eq(dataset)]
        database_count = int(dataset_rows["database_count"].max())
        for family_index, family in enumerate(families):
            family_rows = dataset_rows[dataset_rows["tokenizer_family"].eq(family)].set_index(
                "format"
            )
            present_formats = [value for value in formats if value in family_rows.index]
            if not present_formats:
                continue
            positions = [
                formats.index(value) + offsets[family_index] for value in present_formats
            ]
            medians = [
                float(family_rows.loc[value, "median_overhead_pct"])
                for value in present_formats
            ]
            lower = [
                float(family_rows.loc[value, "median_overhead_pct_low"])
                for value in present_formats
            ]
            upper = [
                float(family_rows.loc[value, "median_overhead_pct_high"])
                for value in present_formats
            ]
            lower_errors = [median - bound for median, bound in zip(medians, lower, strict=True)]
            upper_errors = [bound - median for median, bound in zip(medians, upper, strict=True)]
            axis.errorbar(
                positions,
                medians,
                yerr=[lower_errors, upper_errors],
                fmt="o",
                markersize=7,
                capsize=4,
                linewidth=1.8,
                color=TOKENIZER_FAMILY_COLORS.get(family),
                label=TOKENIZER_FAMILY_LABELS.get(family, family),
            )
        axis.axhline(0.0, color="#444444", linewidth=1.0)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(
            f"{DATASET_LABELS.get(dataset, dataset)} "
            f"({number(database_count, 0)} databases per format)",
            fontweight="bold",
        )
        axis.yaxis.set_major_formatter(number_formatter())

    axes[-1, 0].set_xticks(
        x_positions,
        [FORMAT_LABELS.get(value, value) for value in formats],
    )
    axes[-1, 0].set_xlabel("Serialisation format")
    figure.supylabel("Excess tokens over best tokenizer (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Tokenizer family",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=max(1, len(families)),
        frameon=False,
    )
    tight_layout(figure, rect=(0.04, 0, 1.0, 0.88))
    return figure


def generate_tokenizer_comparison_figure(report: pd.DataFrame, output_dir: Path) -> Path:
    """Calculate, plot, and save the tokenizer comparison, returning its path."""
    output_path = output_dir / TOKENIZER_COMPARISON_FIGURE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_tokenizer_comparison(calculate_tokenizer_overhead(report))

    import matplotlib.pyplot as plt

    save_pgf_figure(figure, output_path)
    plt.close(figure)
    return output_path
