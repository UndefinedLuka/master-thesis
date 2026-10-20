"""Aggregate Text-to-SQL error-correction outcomes and retry-time overhead."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from visualizations.config import DATASET_ORDER, RETRY_OUTCOME_ORDER
from visualizations.text_to_sql_data import TextToSqlAnalysis
from visualizations.uncertainty import mean_intervals


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
    """Calculate terminal attempt counts and exact retry model time."""
    data = analysis.records.copy()
    required = {
        "dataset",
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
    outcome_summary = (
        data.groupby(["dataset", "terminal_outcome"], as_index=False, observed=True)
        .agg(
            run_count=("example_id", "size"),
            total_retry_generation_s=("retry_generation_overhead_s", "sum"),
            average_retry_generation_s=("retry_generation_overhead_s", "mean"),
        )
        .reset_index(drop=True)
    )
    totals = outcome_summary.groupby("dataset", observed=True)["run_count"].transform("sum")
    outcome_summary["run_percentage"] = outcome_summary["run_count"] / totals * 100.0
    outcome_summary = _complete_outcome_grid(outcome_summary)

    stage_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_rows = data[data["dataset"].eq(dataset)]
        for retry_index in (1, 2, 3):
            durations = dataset_rows["attempt_generation_times_s"].map(
                lambda values, index=retry_index: (
                    values[index] if len(values) > index else None
                )
            )
            executed = durations.dropna()
            stage_rows.append(
                {
                    "dataset": dataset,
                    "retry_index": retry_index,
                    "retry_label": f"Retry {retry_index}",
                    "run_count": len(executed),
                    "total_generation_s": float(executed.sum()),
                    "average_generation_s_when_used": (
                        float(executed.mean()) if not executed.empty else 0.0
                    ),
                    "average_generation_s_per_all_runs": (
                        float(executed.sum()) / len(dataset_rows) if len(dataset_rows) else 0.0
                    ),
                }
            )
    retry_stage_summary = pd.DataFrame(stage_rows)
    data["execution_gain_pp"] = (
        data["execution_success"].astype(bool) & data["attempt_count"].gt(1)
    ).astype(float) * 100
    metrics = {
        "execution_gain_pp": "execution_gain_pp",
        "added_seconds": "retry_generation_overhead_s",
    }
    gain_summary = data.groupby("dataset", as_index=False).agg(
        **{name: (source, "mean") for name, source in metrics.items()}
    )
    gain_summary = gain_summary.merge(
        mean_intervals(data, ["dataset"], metrics), on="dataset", validate="one_to_one"
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
    index = pd.MultiIndex.from_product(
        (DATASET_ORDER, RETRY_OUTCOME_ORDER),
        names=("dataset", "terminal_outcome"),
    )
    completed = summary.set_index(["dataset", "terminal_outcome"]).reindex(index)
    completed["run_count"] = completed["run_count"].fillna(0).astype(int)
    for column in (
        "total_retry_generation_s",
        "average_retry_generation_s",
        "run_percentage",
    ):
        completed[column] = completed[column].fillna(0.0)
    return completed.reset_index()
