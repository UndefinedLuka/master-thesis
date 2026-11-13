"""Recorded operational costs on the explicitly selected shared context-fit workload."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from visualizations.config import (
    COMMON_COHORT_METHOD_ORDER,
    COMMON_COHORT_SQL_METHOD,
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.direct_efficiency_data import (
    DIRECT_EFFICIENCY_COLUMNS,
    calculate_direct_efficiency,
    validate_operational_pairing,
)
from visualizations.experiment_outcomes import model_is_excluded
from visualizations.text_to_sql_data import TEXT_TO_SQL_COLUMNS, calculate_text_to_sql_analysis
from visualizations.uncertainty import mean_intervals

COMMON_COHORT_COLUMNS = tuple(dict.fromkeys((*DIRECT_EFFICIENCY_COLUMNS, *TEXT_TO_SQL_COLUMNS)))
PAIR_KEYS = ["dataset", "example_id", "model_id", "db_id"]
SUMMARY_METRICS: dict[str, str] = {
    "average_input_tokens": "input_tokens",
    "average_output_tokens": "output_tokens",
    "average_ttft_s": "cumulative_ttft_s",
    "average_generation_time_s": "generation_after_ttft_s",
    "average_sql_execution_time_s": "query_execution_time_s",
    "average_total_latency_s": "total_time_s",
    "average_retries": "retry_count",
}


@dataclass(frozen=True, slots=True)
class CommonCohortAnalysis:
    """One shared workload, absolute means, and matched model/question comparisons."""

    records: pd.DataFrame
    summary: pd.DataFrame
    paired_comparisons: pd.DataFrame
    question_counts: dict[str, int]
    database_counts: dict[str, int]
    model_count: int


def calculate_common_cohort_analysis(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> CommonCohortAnalysis:
    """Calculate costs for supplied context-fit records, including unsuccessful SQL runs.

    Selection belongs to the shared cohort selector. Validate identical question/model
    sets across six direct formats and SQL without filtering on answers or execution.
    Small synthetic model rosters are supported; campaign selection enforces ten models.
    """
    missing = set(COMMON_COHORT_COLUMNS).difference(results.columns)
    if missing:
        raise ValueError(f"Common-cohort data is missing: {', '.join(sorted(missing))}")
    selected = results[
        results["dataset"].isin(DATASET_ORDER)
        & (
            (results["pipeline"].eq("direct") & results["format"].isin(SERIALIZATION_FORMAT_ORDER))
            | results["pipeline"].eq(COMMON_COHORT_SQL_METHOD)
        )
    ].copy()
    excluded = selected["model_id"].map(
        lambda model_id: model_is_excluded(model_id, excluded_model_markers)
    )
    selected = selected.loc[~excluded]
    if selected.empty:
        raise ValueError("No retained records for the common-cohort comparison")
    is_sql = selected["pipeline"].eq(COMMON_COHORT_SQL_METHOD)
    # SQL is one baseline per question/model, independent of direct serialization.
    selected.loc[is_sql, "format"] = None
    selected["method"] = selected["format"].where(~is_sql, COMMON_COHORT_SQL_METHOD)
    validate_operational_pairing(
        selected, method_column="method", methods=COMMON_COHORT_METHOD_ORDER
    )

    # Both calculators consume this selected workload without reselecting questions.
    calculate_direct_efficiency(selected, excluded_model_markers=())
    sql_analysis = calculate_text_to_sql_analysis(selected, excluded_model_markers=())
    direct = selected[selected["pipeline"].eq("direct")].copy()
    direct["cumulative_ttft_s"] = pd.to_numeric(direct["time_to_first_token_s"])
    direct["generation_after_ttft_s"] = (
        pd.to_numeric(direct["total_time_s"]) - direct["cumulative_ttft_s"]
    ).clip(lower=0)
    direct["query_execution_time_s"] = 0.0
    direct["retry_count"] = 0
    data = pd.concat([direct, sql_analysis.records], ignore_index=True)
    validate_operational_pairing(
        data, method_column="method", methods=COMMON_COHORT_METHOD_ORDER
    )
    for column in SUMMARY_METRICS.values():
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Common-cohort metric {column} must be finite")
    if data["total_time_s"].le(0).any() or data["input_tokens"].le(0).any():
        raise ValueError("Common-cohort latency and input tokens must be positive")

    aggregations = {name: (column, "mean") for name, column in SUMMARY_METRICS.items()}
    aggregations["observation_count"] = ("example_id", "size")
    summary = data.groupby(["dataset", "method"], as_index=False).agg(**aggregations)
    summary = summary.merge(
        mean_intervals(
            data, ["dataset"], SUMMARY_METRICS, comparison="method",
            reference=COMMON_COHORT_SQL_METHOD, effect="ratio",
            pair_keys=("db_id", "example_id", "model_id"),
            min_databases=3,
        ), on=["dataset", "method"], validate="one_to_one",
    )
    sql_records = data[data["method"].eq(COMMON_COHORT_SQL_METHOD)]
    paired = data[data["pipeline"].eq("direct")].merge(
        sql_records[[*PAIR_KEYS, "input_tokens", "output_tokens", "total_time_s"]],
        on=PAIR_KEYS,
        suffixes=("", "_sql"),
        validate="many_to_one",
    )
    paired["latency_ratio"] = paired["total_time_s"] / paired["total_time_s_sql"]
    paired["input_token_ratio"] = paired["input_tokens"] / paired["input_tokens_sql"]
    paired["sql_faster"] = paired["total_time_s_sql"].lt(paired["total_time_s"])
    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    method_rank = {value: index for index, value in enumerate(COMMON_COHORT_METHOD_ORDER)}
    summary = summary.sort_values(
        ["dataset", "method"],
        key=lambda values: values.map(dataset_rank if values.name == "dataset" else method_rank),
    ).reset_index(drop=True)
    return CommonCohortAnalysis(
        records=data,
        summary=summary,
        paired_comparisons=paired,
        question_counts=data.groupby("dataset")["example_id"].nunique().to_dict(),
        database_counts=data.groupby("dataset")["db_id"].nunique().to_dict(),
        model_count=int(data["model_id"].nunique()),
    )
