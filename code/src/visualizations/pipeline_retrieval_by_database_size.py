"""Pooled retrieval comparisons within fixed, database-level payload-size tiers.

The caller selects answerable questions. Sizes use the complete serialization-token
matrix, independently of models, questions, and output success; no inference is run.
"""

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
    DIRECT_ACCURACY_METRIC_ORDER,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.pipeline_retrieval_comparison import (
    PAIR_KEYS,
    SCORE_COLUMNS,
    calculate_pipeline_retrieval_comparison,
)
from visualizations.serialization_data import require_columns
from visualizations.uncertainty import mean_intervals

SIZE_TIER_ORDER: tuple[str, ...] = ("small", "medium", "large")
SIZE_TIER_LABELS: dict[str, str] = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}


@dataclass(frozen=True, slots=True)
class PipelineSizeRetrievalAnalysis:
    """Fixed database sizes, unique runs, pooled tier comparisons, and coverage."""

    database_sizes: pd.DataFrame
    records: pd.DataFrame
    summary: pd.DataFrame
    coverage: pd.DataFrame


def calculate_database_size_tiers(serialization_report: pd.DataFrame) -> pd.DataFrame:
    """Average exactly six formats by three tokenizer families for each database.

    Small payloads average at most 1,500 tokens, medium payloads average more than
    1,500 and at most 10,000, and large payloads average more than 10,000. Every
    supplied database needs the complete 18-record matrix, but dataset/database
    counts and optional serialization metadata are not constrained.
    """
    identity = ["dataset", "database_id", "format", "tokenizer_family"]
    require_columns(serialization_report, (*identity, "tokens"))
    data = serialization_report.loc[:, [*identity, "tokens"]].copy()
    if data.empty:
        raise ValueError("Serialization-token report is empty")
    if data[identity].isna().any(axis=None):
        raise ValueError("Missing serialization-token identity")
    if not data["dataset"].isin(DATASET_ORDER).all():
        raise ValueError("Unsupported dataset in serialization-token report")
    if not data["format"].isin(SERIALIZATION_FORMAT_ORDER).all():
        raise ValueError("Unsupported serialization format in token report")
    if not data["tokenizer_family"].isin(TOKENIZER_FAMILY_ORDER).all():
        raise ValueError("Unsupported tokenizer family in token report")
    if data.duplicated(identity).any():
        raise ValueError("Duplicate serialization-token identity")
    tokens = pd.to_numeric(data["tokens"], errors="coerce")
    if tokens.isna().any() or not (np.isfinite(tokens) & tokens.gt(0)).all():
        raise ValueError("Serialization token counts must be numeric, positive, and finite")
    data["tokens"] = tokens

    # Supported identities plus uniqueness make 18 rows equivalent to the complete
    # Cartesian product, without imposing the campaign's full database counts.
    counts = data.groupby(["dataset", "database_id"], observed=True).size()
    incomplete = counts.ne(len(SERIALIZATION_FORMAT_ORDER) * len(TOKENIZER_FAMILY_ORDER))
    if incomplete.any():
        dataset, database_id = counts.index[incomplete][0]
        raise ValueError(
            f"Incomplete serialization-token coverage for {dataset}/{database_id}: "
            "all six formats and three tokenizer families are required"
        )
    sizes = data.groupby(["dataset", "database_id"], as_index=False, observed=True).agg(
        mean_serialized_tokens=("tokens", "mean"),
    ).rename(columns={"database_id": "db_id"})
    sizes["size_tier"] = np.select(
        [sizes["mean_serialized_tokens"].le(1_500),
         sizes["mean_serialized_tokens"].le(10_000)],
        ["small", "medium"], default="large",
    )
    return sizes.sort_values(
        ["dataset", "db_id"],
        key=lambda values: values.map({v: i for i, v in enumerate(DATASET_ORDER)})
        if values.name == "dataset" else values,
    ).reset_index(drop=True)


def calculate_pipeline_size_retrieval_comparison(
    results: pd.DataFrame,
    serialization_report: pd.DataFrame,
    *,
    excluded_model_markers: Sequence[str] = DEFAULT_EXCLUDED_MODEL_MARKERS,
) -> PipelineSizeRetrievalAnalysis:
    """Compare all six direct formats with SQL, pooling datasets within each tier.

    The supplied cohort must already contain only answerable questions. The shared
    retrieval validator applies model exclusions, requires identical paired run
    identities, and retains failures with explicit zero scores. Means give every
    retained run equal weight. Pointwise 95% database-bootstrap intervals and paired
    direct-minus-SQL effects resample distinct dataset/database clusters within each
    pooled size tier, without stratifying by dataset. Database and question counts
    use dataset-qualified identities. Single-database tiers have no estimable
    between-database interval; returned records retain their original identities.
    """
    database_sizes = calculate_database_size_tiers(serialization_report)
    validated = calculate_pipeline_retrieval_comparison(
        results, excluded_model_markers=excluded_model_markers,
    )
    records = validated.records.drop(
        columns=["mean_serialized_tokens", "size_tier"], errors="ignore",
    ).merge(database_sizes, on=["dataset", "db_id"], how="left", validate="many_to_one")
    missing_size = records["mean_serialized_tokens"].isna()
    if missing_size.any():
        missing = records.loc[missing_size, ["dataset", "db_id"]].iloc[0].to_dict()
        raise ValueError(f"Missing serialization size for database: {missing}")

    # Only the temporary analysis frame changes db_id. The shared bootstrap helper
    # clusters on that column, so dataset qualification prevents cross-dataset
    # collisions while keeping dataset in the explicit pairing keys.
    pooled = records.assign(
        db_id=list(zip(records["dataset"], records["db_id"], strict=True)),
        _question_identity=list(zip(records["dataset"], records["example_id"], strict=True)),
    )
    long = pooled.melt(
        id_vars=[*PAIR_KEYS, "size_tier", "method", "_question_identity"],
        value_vars=list(SCORE_COLUMNS.values()), var_name="metric", value_name="score",
    )
    long["metric"] = long["metric"].map(
        {source: metric for metric, source in SCORE_COLUMNS.items()}
    )
    summary = long.groupby(
        ["size_tier", "method", "metric"], as_index=False, observed=True,
    ).agg(
        mean_score=("score", "mean"), observation_count=("score", "size"),
        question_count=("_question_identity", "nunique"), model_count=("model_id", "nunique"),
        database_count=("db_id", "nunique"),
    )
    intervals = mean_intervals(
        long, ["size_tier", "metric"], {"mean_score": "score"},
        comparison="method", reference=COMMON_COHORT_SQL_METHOD, pair_keys=PAIR_KEYS,
    )
    summary = summary.merge(
        intervals, on=["size_tier", "metric", "method"], validate="one_to_one",
    )
    order = {
        "size_tier": SIZE_TIER_ORDER,
        "metric": DIRECT_ACCURACY_METRIC_ORDER, "method": COMMON_COHORT_METHOD_ORDER,
    }
    summary = summary.sort_values(
        list(order),
        key=lambda values: values.map({v: i for i, v in enumerate(order[values.name])}),
    ).reset_index(drop=True)

    # Pairing has been validated, so SQL represents each question/model exactly
    # once. Counting all methods would inflate observation coverage sevenfold.
    paired = pooled.loc[pooled["method"].eq(COMMON_COHORT_SQL_METHOD)]
    coverage = paired.groupby("size_tier", observed=True).agg(
        database_count=("db_id", "nunique"),
        question_count=("_question_identity", "nunique"),
        model_count=("model_id", "nunique"), observation_count=("model_id", "size"),
    ).reindex(
        pd.Index(SIZE_TIER_ORDER, name="size_tier"), fill_value=0,
    ).reset_index()
    return PipelineSizeRetrievalAnalysis(
        database_sizes=database_sizes, records=records.reset_index(drop=True),
        summary=summary, coverage=coverage,
    )
