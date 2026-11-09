"""Shared measurement, validation, and ordering helpers for serialization analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from config.paths import ProjectPaths
from serialization.tokens import measure_serialization_tokens
from visualizations.config import (
    DATASET_ORDER,
    EXPECTED_DATABASE_COUNTS,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_ORDER,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset",
    "database_id",
    "format",
    "tokenizer_family",
    "tokenizer_label",
    "tokenizer_model",
    "tokens",
)
IDENTITY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "database_id",
    "format",
    "tokenizer_family",
)


@dataclass(frozen=True, slots=True)
class SerializationCoverageStats:
    """Dimensions of a complete serialization-token report."""

    database_counts: Mapping[str, int]
    format_count: int
    tokenizer_family_count: int
    record_count: int


def measure_token_report(
    paths: ProjectPaths | None = None,
    *,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the analysis input directly from databases and GGUFs, without a disk cache."""
    records = measure_serialization_tokens(paths, models_dir=models_dir)
    return validate_token_report(pd.DataFrame(asdict(record) for record in records))


def validate_token_report(report: pd.DataFrame) -> pd.DataFrame:
    """Validate in-memory token measurements and return a normalized copy."""
    require_columns(report, REQUIRED_COLUMNS)
    report = report.copy()
    report["tokens"] = pd.to_numeric(report["tokens"], errors="coerce")
    if report["tokens"].isna().any():
        raise ValueError("Serialization-token report contains a missing or non-numeric token count")
    if report["tokens"].le(0).any():
        raise ValueError("Serialization-token report contains a non-positive token count")
    return report


def validate_complete_coverage(
    report: pd.DataFrame,
    *,
    expected_database_counts: Mapping[str, int] = EXPECTED_DATABASE_COUNTS,
    expected_formats: Sequence[str] = SERIALIZATION_FORMAT_ORDER,
    expected_families: Sequence[str] = TOKENIZER_FAMILY_ORDER,
) -> SerializationCoverageStats:
    """Validate the complete dataset × database × format × tokenizer-family matrix."""
    require_columns(report, REQUIRED_COLUMNS)
    if report.empty:
        raise ValueError("Serialization-token report is empty")
    if report[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Serialization-token report contains missing required values")

    duplicate_mask = report.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicate = report.loc[duplicate_mask, list(IDENTITY_COLUMNS)].iloc[0].to_dict()
        raise ValueError(f"Duplicate serialization-token identity: {duplicate}")

    actual_datasets = set(report["dataset"].astype(str))
    expected_datasets = set(expected_database_counts)
    if actual_datasets != expected_datasets:
        raise ValueError(
            f"Dataset coverage differs: expected {sorted(expected_datasets)}, "
            f"found {sorted(actual_datasets)}"
        )

    actual_formats = set(report["format"].astype(str))
    if actual_formats != set(expected_formats):
        raise ValueError(
            f"Format coverage differs: expected {sorted(expected_formats)}, "
            f"found {sorted(actual_formats)}"
        )

    actual_families = set(report["tokenizer_family"].astype(str))
    if actual_families != set(expected_families):
        raise ValueError(
            f"Tokenizer-family coverage differs: expected {sorted(expected_families)}, "
            f"found {sorted(actual_families)}"
        )

    labels_per_family = report.groupby("tokenizer_family", observed=True)[
        "tokenizer_label"
    ].nunique()
    if labels_per_family.ne(1).any():
        family = str(labels_per_family[labels_per_family.ne(1)].index[0])
        raise ValueError(f"Tokenizer family {family!r} has multiple display labels")

    database_counts: dict[str, int] = {}
    expected_combinations = len(expected_formats) * len(expected_families)
    expected_identity_set = {
        (serialization_format, family)
        for serialization_format in expected_formats
        for family in expected_families
    }
    for dataset, expected_count in expected_database_counts.items():
        dataset_rows = report[report["dataset"].eq(dataset)]
        databases = tuple(sorted(dataset_rows["database_id"].astype(str).unique()))
        if len(databases) != expected_count:
            raise ValueError(
                f"Expected {expected_count} databases for {dataset}, found {len(databases)}"
            )
        database_counts[dataset] = len(databases)

        coverage = dataset_rows.groupby("database_id", observed=True).apply(
            lambda rows: frozenset(
                zip(rows["format"].astype(str), rows["tokenizer_family"].astype(str), strict=True)
            ),
            include_groups=False,
        )
        incomplete = coverage[
            coverage.map(lambda combinations: combinations != expected_identity_set)
        ]
        if not incomplete.empty:
            database_id = str(incomplete.index[0])
            missing = sorted(expected_identity_set.difference(incomplete.iloc[0]))
            raise ValueError(
                f"Incomplete serialization-token coverage for {dataset}/{database_id}: "
                f"missing {missing}"
            )

        expected_rows = expected_count * expected_combinations
        if len(dataset_rows) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows for {dataset}, found {len(dataset_rows)}"
            )

    expected_total = sum(expected_database_counts.values()) * expected_combinations
    if len(report) != expected_total:
        raise ValueError(f"Expected {expected_total} report rows, found {len(report)}")
    return SerializationCoverageStats(
        database_counts=database_counts,
        format_count=len(actual_formats),
        tokenizer_family_count=len(actual_families),
        record_count=len(report),
    )


def sort_summary(
    summary: pd.DataFrame,
    *,
    family_column: str | None = None,
) -> pd.DataFrame:
    """Sort a summary using publication dataset, format, and optional family orders."""
    data = summary.copy()
    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    format_rank = {value: index for index, value in enumerate(SERIALIZATION_FORMAT_ORDER)}
    data["_dataset_rank"] = data["dataset"].map(dataset_rank).fillna(len(dataset_rank))
    data["_format_rank"] = data["format"].map(format_rank).fillna(len(format_rank))
    sort_columns = ["_dataset_rank", "_format_rank"]
    if family_column is not None:
        family_rank = {value: index for index, value in enumerate(TOKENIZER_FAMILY_ORDER)}
        data["_family_rank"] = data[family_column].map(family_rank).fillna(len(family_rank))
        sort_columns.append("_family_rank")
    return data.sort_values(sort_columns).drop(columns=sort_columns).reset_index(drop=True)


def present_order(preferred: Sequence[str], values: pd.Series) -> list[str]:
    """Restrict a preferred order to values present in a series, appending unknown values."""
    present = {str(value) for value in values.dropna().unique()}
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when an analysis input lacks required columns."""
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Report data is missing required column(s): {', '.join(sorted(missing))}")
