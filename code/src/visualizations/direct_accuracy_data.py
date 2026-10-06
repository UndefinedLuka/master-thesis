"""Shared preparation and validation for direct-context retrieval-accuracy figures."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

from visualizations.config import (
    DATASET_ORDER,
    DEFAULT_EXCLUDED_MODEL_MARKERS,
    DIRECT_ACCURACY_METRIC_ORDER,
    SERIALIZATION_FORMAT_ORDER,
)
from visualizations.experiment_outcomes import model_is_excluded
from visualizations.uncertainty import mean_intervals

METRIC_SOURCE_COLUMNS: dict[str, str] = {
    "normalized_exact_match": "scores.normalized_exact_match",
    "bertscore_f": "scores.bertscore_f",
    "rouge_l_f": "scores.rouge_l_f",
}
DIRECT_ACCURACY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "db_id",
    "pipeline",
    "format",
    "model_id",
    "example_id",
    *METRIC_SOURCE_COLUMNS.values(),
)
MODEL_FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "qwen-3.5": re.compile(r"^Qwen3\.5-(?P<size>\d+(?:\.\d+)?)B", re.IGNORECASE),
    "ministral-3": re.compile(
        r"^Ministral-3-(?P<size>\d+(?:\.\d+)?)B",
        re.IGNORECASE,
    ),
    "gemma-4": re.compile(r"^gemma-4-(?:E)?(?P<size>\d+(?:\.\d+)?)B", re.IGNORECASE),
}


def validate_direct_accuracy_scores(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> int:
    """Validate that every retained direct run has all three retrieval-accuracy scores."""
    _require_columns(results, DIRECT_ACCURACY_COLUMNS)
    direct = _retained_direct_rows(results, excluded_model_markers=excluded_model_markers)
    if direct.empty:
        raise ValueError("No retained direct-context result records are available")

    for metric, source_column in METRIC_SOURCE_COLUMNS.items():
        numeric = pd.to_numeric(direct[source_column], errors="coerce")
        missing_count = int(numeric.isna().sum())
        if missing_count:
            raise ValueError(
                f"Direct-context accuracy has {missing_count:,} missing {metric} score(s)"
            )
    return len(direct)


def prepare_direct_accuracy(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> pd.DataFrame:
    """Return direct-only scores in long form, retaining explicit zero-scored failures."""
    _require_columns(results, DIRECT_ACCURACY_COLUMNS)
    direct = _retained_direct_rows(results, excluded_model_markers=excluded_model_markers)
    renamed = direct.rename(
        columns={source: metric for metric, source in METRIC_SOURCE_COLUMNS.items()}
    )
    long_data = renamed.melt(
        id_vars=("dataset", "db_id", "model_id", "example_id", "format"),
        value_vars=DIRECT_ACCURACY_METRIC_ORDER,
        var_name="metric",
        value_name="score",
    )
    long_data["score"] = pd.to_numeric(long_data["score"], errors="coerce")
    return long_data.dropna(subset=["score"]).reset_index(drop=True)


def calculate_format_accuracy(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
    include_sql: bool = False,
) -> pd.DataFrame:
    """Calculate mean direct-context score for each dataset, format, and metric."""
    data = prepare_direct_accuracy(
        results,
        excluded_model_markers=excluded_model_markers,
    )
    if include_sql:
        sql = results[results["pipeline"].eq("text_to_sql")].copy()
        sql = sql.loc[~sql["model_id"].map(
            lambda model: model_is_excluded(model, excluded_model_markers)
        )]
        sql["format"] = "text_to_sql"
        sql = sql.rename(
            columns={source: metric for metric, source in METRIC_SOURCE_COLUMNS.items()}
        )
        sql = sql.melt(
            id_vars=("dataset", "db_id", "model_id", "example_id", "format"),
            value_vars=DIRECT_ACCURACY_METRIC_ORDER, var_name="metric", value_name="score",
        )
        data = pd.concat([data, sql], ignore_index=True)
    if data.empty:
        return _empty_format_summary()
    summary = (
        data.groupby(["dataset", "format", "metric"], as_index=False, observed=True)
        .agg(
            mean_score=("score", "mean"),
            observation_count=("score", "size"),
            model_count=("model_id", "nunique"),
            question_count=("example_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    intervals = mean_intervals(
        data, ["dataset", "metric"], {"mean_score": "score"},
        comparison="format", reference="csv",
        pair_keys=("db_id", "example_id", "model_id"),
    )
    return _sort_accuracy_summary(summary.merge(
        intervals, on=["dataset", "metric", "format"], validate="one_to_one"
    ))


def calculate_model_size_accuracy(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> pd.DataFrame:
    """Calculate per-model scores averaged equally across all direct serialization runs."""
    data = prepare_direct_accuracy(
        results,
        excluded_model_markers=excluded_model_markers,
    )
    if data.empty:
        return _empty_model_size_summary()

    metadata = data["model_id"].map(parse_model_family_size)
    data = data[metadata.notna()].copy()
    if data.empty:
        return _empty_model_size_summary()
    retained_metadata = metadata[metadata.notna()]
    data["model_family"] = [value[0] for value in retained_metadata]
    data["model_size_b"] = [value[1] for value in retained_metadata]
    data["size_label"] = data["model_id"].map(model_size_label)

    summary = (
        data.groupby(
            [
                "dataset",
                "model_id",
                "model_family",
                "model_size_b",
                "size_label",
                "metric",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_score=("score", "mean"),
            observation_count=("score", "size"),
            format_count=("format", "nunique"),
            question_count=("example_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    intervals = mean_intervals(
        data, ["dataset", "metric", "model_family"], {"mean_score": "score"},
        comparison="model_id", pair_keys=("db_id", "example_id", "format"),
    )
    summary = summary.merge(
        intervals, on=["dataset", "metric", "model_family", "model_id"], validate="one_to_one"
    )
    return summary.sort_values(
        ["dataset", "metric", "model_family", "model_size_b", "model_id"]
    ).reset_index(drop=True)


def parse_model_family_size(model_id: object) -> tuple[str, float] | None:
    """Parse a supported model family and nominal parameter count in billions."""
    text = str(model_id)
    for family, pattern in MODEL_FAMILY_PATTERNS.items():
        match = pattern.search(text)
        if match:
            return family, float(match.group("size"))
    return None


def model_size_label(model_id: object) -> str:
    """Return a concise size label, including active parameters for mixture models."""
    text = str(model_id)
    metadata = parse_model_family_size(text)
    if metadata is None:
        return text
    _, size = metadata
    label = _format_size(size)
    active_match = re.search(r"-A(?P<active>\d+(?:\.\d+)?)B", text, re.IGNORECASE)
    if active_match:
        label += f" (A{active_match.group('active')} active)"
    return label


def _retained_direct_rows(
    results: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str],
) -> pd.DataFrame:
    selected = results[
        results["dataset"].isin(DATASET_ORDER)
        & results["pipeline"].eq("direct")
        & results["format"].isin(SERIALIZATION_FORMAT_ORDER)
    ].copy()
    excluded = selected["model_id"].map(
        lambda model_id: model_is_excluded(model_id, excluded_model_markers)
    )
    return selected.loc[~excluded]


def _sort_accuracy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    format_rank = {
        value: index for index, value in enumerate(SERIALIZATION_FORMAT_ORDER)
    }
    metric_rank = {
        value: index for index, value in enumerate(DIRECT_ACCURACY_METRIC_ORDER)
    }
    data["_dataset_rank"] = data["dataset"].map(dataset_rank)
    data["_format_rank"] = data["format"].map(format_rank)
    data["_metric_rank"] = data["metric"].map(metric_rank)
    return data.sort_values(
        ["_dataset_rank", "_metric_rank", "_format_rank"]
    ).drop(columns=["_dataset_rank", "_format_rank", "_metric_rank"]).reset_index(drop=True)


def _format_size(size: float) -> str:
    return str(int(size)) if size.is_integer() else f"{size:g}"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(
            f"Direct-context accuracy data is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )


def _empty_format_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "dataset",
            "format",
            "metric",
            "mean_score",
            "observation_count",
            "model_count",
            "question_count",
        )
    )


def _empty_model_size_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "dataset",
            "model_id",
            "model_family",
            "model_size_b",
            "size_label",
            "metric",
            "mean_score",
            "observation_count",
            "format_count",
            "question_count",
        )
    )
