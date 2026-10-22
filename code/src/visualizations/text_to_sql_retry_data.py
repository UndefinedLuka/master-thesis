"""Aggregate Text-to-SQL error-correction outcomes and retry-time overhead."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from visualizations.config import DATASET_ORDER, RETRY_OUTCOME_ORDER
from visualizations.text_to_sql_data import TextToSqlAnalysis
from visualizations.uncertainty import bounds, database_weights, mean_intervals


@dataclass(frozen=True, slots=True)
class TextToSqlRetryAnalysis:
    """Terminal attempt outcomes and generation overhead introduced by retries."""

    outcome_summary: pd.DataFrame
    retry_stage_summary: pd.DataFrame
    run_count: int
    retried_run_count: int
    total_retry_generation_s: float
    gain_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def calculate_text_to_sql_retry_analysis(
    analysis: TextToSqlAnalysis,
) -> TextToSqlRetryAnalysis:
    """Summarize the caller's cohort, including terminal failures and their work."""
    data = analysis.records.copy()
    required = {
        "dataset",
        "db_id",
        "example_id",
        "execution_success",
        "attempt_count",
        "attempt_generation_times_s",
        "retry_generation_overhead_s",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Text-to-SQL retry data is missing: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("No Text-to-SQL retry records are available")

    data["terminal_outcome"] = data.apply(_terminal_outcome, axis=1)
    data["first_attempt_success_rate"] = (
        data["execution_success"] & data["attempt_count"].eq(1)
    ).astype(float)
    data["final_success_rate"] = data["execution_success"].astype(float)
    data["initially_failed"] = data["first_attempt_success_rate"].eq(0)
    data["entered_correction"] = data["attempt_count"].gt(1)
    data["retries"] = data["attempt_count"] - 1
    outcome_summary = (
        data.groupby(["dataset", "terminal_outcome"], as_index=False, observed=True)
        .agg(
            run_count=("example_id", "size"),
            total_retry_generation_s=("retry_generation_overhead_s", "sum"),
            average_retry_generation_s=("retry_generation_overhead_s", "mean"),
            total_retries=("retries", "sum"),
            average_retries=("retries", "mean"),
        )
        .reset_index(drop=True)
    )
    totals = outcome_summary.groupby("dataset", observed=True)["run_count"].transform("sum")
    outcome_summary["run_percentage"] = outcome_summary["run_count"] / totals * 100.0
    outcome_summary = _complete_outcome_grid(outcome_summary)

    stage_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_rows = data[data["dataset"].eq(dataset)]
        if dataset_rows.empty:
            continue
        for retry_index in (1, 2, 3):
            durations = dataset_rows["attempt_generation_times_s"].map(
                lambda values, index=retry_index: (
                    values[index] if len(values) > index else None
                )
            )
            executed = durations.dropna()
            stage = {
                "dataset": dataset,
                "retry_index": retry_index,
                "retry_label": f"Retry {retry_index}",
                "run_count": len(executed),
                "total_generation_s": float(executed.sum()),
            }
            stage_data = dataset_rows.assign(
                stage_generation=durations.fillna(0.0), all_runs=1.0,
                stage_used=durations.notna(),
            )
            for metric, denominator in (
                ("average_generation_s_per_all_runs", "all_runs"),
                ("average_generation_s_per_retried_run", "entered_correction"),
                ("average_generation_s_when_used", "stage_used"),
            ):
                stage.update(_conditional_estimates(
                    stage_data, "stage_generation", denominator, metric
                ))
            stage_rows.append(stage)
    retry_stage_summary = pd.DataFrame(stage_rows)
    data["execution_gain_pp"] = (
        data["execution_success"].astype(bool) & data["attempt_count"].gt(1)
    ).astype(float) * 100
    metrics = {
        "first_attempt_success_rate": "first_attempt_success_rate",
        "final_success_rate": "final_success_rate",
        "execution_gain_pp": "execution_gain_pp",
        "average_retries": "retries",
        "added_seconds": "retry_generation_overhead_s",
    }
    gain_summary = data.groupby("dataset", as_index=False).agg(
        **{name: (source, "mean") for name, source in metrics.items()}
    )
    gain_summary = gain_summary.merge(
        mean_intervals(data, ["dataset"], metrics), on="dataset", validate="one_to_one"
    )
    counts = data.groupby("dataset", as_index=False).agg(
        run_count=("example_id", "size"),
        initially_failed_run_count=("initially_failed", "sum"),
        retried_run_count=("entered_correction", "sum"),
    )
    gain_summary = gain_summary.merge(counts, on="dataset", validate="one_to_one")
    for denominator, conditional_metrics in (
        ("initially_failed", {"recovery_rate": "final_success_rate"}),
        ("entered_correction", {
            "average_retries_per_retried_run": "retries",
            "added_seconds_per_retried_run": "retry_generation_overhead_s",
        }),
    ):
        conditional_rows = []
        for dataset, group in data.groupby("dataset", observed=True):
            row = {"dataset": dataset}
            for metric, source in conditional_metrics.items():
                row.update(_conditional_estimates(group, source, denominator, metric))
            conditional_rows.append(row)
        conditional = pd.DataFrame(conditional_rows)
        gain_summary = gain_summary.merge(
            conditional, on="dataset", how="left", validate="one_to_one"
        )
    return TextToSqlRetryAnalysis(
        outcome_summary=outcome_summary,
        retry_stage_summary=retry_stage_summary,
        run_count=len(data),
        retried_run_count=int(data["attempt_count"].gt(1).sum()),
        total_retry_generation_s=float(data["retry_generation_overhead_s"].sum()),
        gain_summary=gain_summary,
    )


def _terminal_outcome(row: pd.Series) -> str:
    if not bool(row["execution_success"]):
        return "failure"
    attempt_count = int(row["attempt_count"])
    if not 1 <= attempt_count <= 4:
        raise ValueError(f"Unexpected successful attempt count: {attempt_count}")
    return f"attempt_{attempt_count}"


def _complete_outcome_grid(summary: pd.DataFrame) -> pd.DataFrame:
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(summary["dataset"])]
    index = pd.MultiIndex.from_product(
        (datasets, RETRY_OUTCOME_ORDER),
        names=("dataset", "terminal_outcome"),
    )
    completed = summary.set_index(["dataset", "terminal_outcome"]).reindex(index)
    completed["run_count"] = completed["run_count"].fillna(0).astype(int)
    for column in (
        "total_retry_generation_s",
        "total_retries",
        "run_percentage",
    ):
        completed[column] = completed[column].fillna(0.0)
    return completed.reset_index()


def _conditional_estimates(
    data: pd.DataFrame, source: str, denominator: str, metric: str,
) -> dict[str, float]:
    """Bootstrap a conditional ratio over every eligible database cluster.

    Zero-denominator clusters remain in the resampling universe. If any draw has
    no denominator, the interval is not estimable, matching the shared bounds
    convention; such draws are never silently discarded.
    """
    eligible = data[denominator].astype(float)
    aggregate = data.assign(
        _numerator=data[source].astype(float).where(eligible.gt(0), 0.0),
        _denominator=eligible,
    ).groupby("db_id", sort=True)[["_numerator", "_denominator"]].sum()
    total = float(aggregate["_denominator"].sum())
    estimate = float(aggregate["_numerator"].sum()) / total if total else float("nan")
    weights = database_weights(len(aggregate))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = (weights @ aggregate["_numerator"].to_numpy()) / (
            weights @ aggregate["_denominator"].to_numpy()
        )
    low, high = bounds(draws) if len(aggregate) > 1 else (float("nan"), float("nan"))
    return {metric: estimate, f"{metric}_low": low, f"{metric}_high": high}
