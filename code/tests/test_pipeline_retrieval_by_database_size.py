"""Regression coverage for paired retrieval accuracy within database-size tiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from visualizations.config import (
    COMMON_COHORT_METHOD_ORDER,
    DIRECT_ACCURACY_METRIC_ORDER,
    SERIALIZATION_FORMAT_ORDER,
    TOKENIZER_FAMILY_ORDER,
)
from visualizations.pipeline_retrieval_by_database_size import (
    SIZE_TIER_LABELS,
    SIZE_TIER_ORDER,
    calculate_database_size_tiers,
    calculate_pipeline_size_retrieval_comparison,
)
from visualizations.pipeline_retrieval_comparison import calculate_pipeline_retrieval_comparison
from visualizations.uncertainty import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED

# The shared database ID deliberately belongs to different tiers in the two datasets.
DATABASES = (
    ("spider", "shared", 1_000, 1, 1),
    ("spider", "small_many", 1_500, 3, 0),
    ("spider", "medium", 5_000, 2, 1),
    ("spider", "large", 12_000, 2, 0),
    ("bird", "shared", 11_000, 1, 1),
    ("bird", "large_many", 20_000, 3, 0),
)
SCORE_COLUMNS = [f"scores.{metric}" for metric in DIRECT_ACCURACY_METRIC_ORDER]


def _token_report(databases: tuple = DATABASES) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset": dataset,
            "database_id": database,
            "format": fmt,
            "tokenizer_family": family,
            "tokens": tokens,
        }
        for dataset, database, tokens, *_ in databases
        for fmt in SERIALIZATION_FORMAT_ORDER
        for family in TOKENIZER_FAMILY_ORDER
    ])


def _matrix(databases: tuple = DATABASES) -> pd.DataFrame:
    rows = []
    for dataset, database, _, questions, direct_score in databases:
        for question in range(questions):
            for model in ("Qwen3.5-2B", "gemma-4-E4B"):
                for method in (*SERIALIZATION_FORMAT_ORDER, "text_to_sql"):
                    sql = method == "text_to_sql"
                    nem = 1 - direct_score if sql else direct_score
                    rows.append({
                        "dataset": dataset,
                        "db_id": database,
                        "example_id": f"{database}:{question}",
                        "model_id": model,
                        "pipeline": "text_to_sql" if sql else "direct",
                        "format": None if sql else method,
                        "error": None,
                        "execution_success": True if sql else None,
                        "has_answer": True,
                        "scores.normalized_exact_match": nem,
                        "scores.bertscore_f": 0.2 + 0.6 * nem,
                        "scores.rouge_l_f": 0.1 + 0.8 * nem,
                    })
    return pd.DataFrame(rows)


def test_tier_boundaries_are_inclusive_and_support_synthetic_database_counts() -> None:
    assert SIZE_TIER_ORDER == ("small", "medium", "large")
    assert set(SIZE_TIER_LABELS) == set(SIZE_TIER_ORDER)
    assert all(isinstance(label, str) and label for label in SIZE_TIER_LABELS.values())
    cases = (
        ("spider", "below_small", 1_499),
        ("spider", "at_small", 1_500),
        ("spider", "above_small", 1_501),
        ("bird", "below_medium", 9_999),
        ("bird", "at_medium", 10_000),
        ("bird", "above_medium", 10_001),
    )
    report = _token_report(cases).sample(frac=1, random_state=3)
    original = report.copy(deep=True)
    sizes = calculate_database_size_tiers(report)
    assert {"dataset", "db_id", "mean_serialized_tokens", "size_tier"} <= set(sizes)
    assert len(sizes) == len(cases)
    assert not sizes.duplicated(["dataset", "db_id"]).any()
    indexed = sizes.set_index(["dataset", "db_id"])
    for (dataset, database, tokens), tier in zip(
        cases, ("small", "small", "medium", "medium", "medium", "large"), strict=True,
    ):
        assert indexed.loc[(dataset, database), "mean_serialized_tokens"] == tokens
        assert indexed.loc[(dataset, database), "size_tier"] == tier
    pd.testing.assert_frame_equal(report, original)


def test_size_is_one_mean_across_all_eighteen_format_tokenizer_measurements() -> None:
    report = _token_report((("spider", "mixed", 1_000),))
    assert len(report) == 18
    report.loc[0, "tokens"] = 19_000
    sizes = calculate_database_size_tiers(report)
    assert len(sizes) == 1
    assert sizes.iloc[0]["mean_serialized_tokens"] == pytest.approx(2_000)
    assert sizes.iloc[0]["size_tier"] == "medium"


@pytest.mark.parametrize("boundary,tier", [(1_500, "medium"), (10_000, "large")])
def test_fractional_mean_is_not_rounded_before_tiering(boundary: int, tier: str) -> None:
    report = _token_report((("spider", "fractional", boundary),))
    report.loc[0, "tokens"] += 1
    row = calculate_database_size_tiers(report).iloc[0]
    assert row["mean_serialized_tokens"] == pytest.approx(boundary + 1 / 18)
    assert row["size_tier"] == tier


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, "invalid", None])
def test_token_counts_must_be_positive_finite_numbers(value: object) -> None:
    report = _token_report()
    report["tokens"] = report["tokens"].astype(object)
    report.loc[0, "tokens"] = value
    with pytest.raises(ValueError):
        calculate_database_size_tiers(report)


@pytest.mark.parametrize("problem", [
    "missing_measurement", "duplicate", "duplicate_replaces_missing", "unknown_format",
    "unknown_family", "null_dataset", "null_database", "null_format", "null_family",
])
def test_token_measurement_identities_must_be_unique_and_complete(problem: str) -> None:
    report = _token_report()
    if problem == "missing_measurement":
        report = report.drop(0)
    elif problem == "duplicate":
        report = pd.concat([report, report.iloc[[0]]], ignore_index=True)
    elif problem == "duplicate_replaces_missing":
        report = pd.concat([report.drop(1), report.iloc[[0]]], ignore_index=True)
    elif problem == "unknown_format":
        report.loc[0, "format"] = "unsupported"
    elif problem == "unknown_family":
        report.loc[0, "tokenizer_family"] = "unsupported"
    else:
        column = {
            "null_dataset": "dataset", "null_database": "database_id",
            "null_format": "format", "null_family": "tokenizer_family",
        }[problem]
        report.loc[0, column] = None
    with pytest.raises(ValueError):
        calculate_database_size_tiers(report)


@pytest.mark.parametrize("column", [
    "dataset", "database_id", "format", "tokenizer_family", "tokens",
])
def test_token_report_requires_identity_and_count_columns(column: str) -> None:
    with pytest.raises(ValueError):
        calculate_database_size_tiers(_token_report().drop(columns=column))


def test_records_have_one_dataset_specific_tier_and_sql_is_not_repeated() -> None:
    data, report = _matrix(), _token_report()
    original_data, original_report = data.copy(deep=True), report.copy(deep=True)
    analysis = calculate_pipeline_size_retrieval_comparison(
        data.sample(frac=1, random_state=7), report.sample(frac=1, random_state=9),
    )
    assert len(analysis.database_sizes) == len(DATABASES)
    assert len(analysis.records) == len(data)
    assert len(analysis.summary) == len(SIZE_TIER_ORDER) * len(COMMON_COHORT_METHOD_ORDER) * 3
    assert "dataset" not in analysis.summary
    assert "dataset" not in analysis.coverage
    identity = ["dataset", "db_id", "example_id", "model_id", "method"]
    assert not analysis.records.duplicated(identity).any()
    sql = analysis.records.loc[analysis.records["method"].eq("text_to_sql")]
    assert len(sql) == len(data.loc[data["pipeline"].eq("text_to_sql")])
    shared = analysis.records.loc[analysis.records["db_id"].eq("shared")]
    assert set(shared.loc[shared["dataset"].eq("spider"), "size_tier"]) == {"small"}
    assert set(shared.loc[shared["dataset"].eq("bird"), "size_tier"]) == {"large"}
    pd.testing.assert_frame_equal(data, original_data)
    pd.testing.assert_frame_equal(report, original_report)


@pytest.mark.parametrize("tier", ["small", "large"])
def test_tier_scores_are_run_weighted_with_paired_database_bootstrap_intervals(tier: str) -> None:
    analysis = calculate_pipeline_size_retrieval_comparison(_matrix(), _token_report())
    # Sorted dataset/database clusters: large pools two BIRD databases and one Spider
    # database in a single multinomial draw, rather than fixing dataset proportions.
    runs, successes = ([2, 6], [1, 0]) if tier == "small" else ([6, 2, 4], [0, 1, 0])
    runs, successes = np.array(runs), np.array(successes)
    database_count = len(runs)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(
        database_count, np.full(database_count, 1 / database_count), size=BOOTSTRAP_DRAWS,
    )
    for metric, (base, span) in zip(
        DIRECT_ACCURACY_METRIC_ORDER, ((0, 1), (0.2, 0.6), (0.1, 0.8)), strict=True,
    ):
        direct_scores = base + span * successes
        sql_scores = base + span * (1 - successes)
        direct_draws = weights @ (runs * direct_scores) / (weights @ runs)
        sql_draws = weights @ (runs * sql_scores) / (weights @ runs)
        direct_mean = np.average(direct_scores, weights=runs)
        sql_mean = np.average(sql_scores, weights=runs)
        panel = analysis.summary.loc[
            analysis.summary["size_tier"].eq(tier)
            & analysis.summary["metric"].eq(metric)
        ].set_index("method")
        assert set(panel.index) == set(COMMON_COHORT_METHOD_ORDER)
        assert panel["observation_count"].eq(runs.sum()).all()
        assert panel["database_count"].eq(database_count).all()
        assert panel["question_count"].eq(runs.sum() // 2).all()
        assert panel["model_count"].eq(2).all()
        assert panel["effect_reference"].eq("text_to_sql").all()
        sql = panel.loc["text_to_sql"]
        assert sql["mean_score"] == pytest.approx(sql_mean)
        assert sql["mean_score_effect"] == 0
        assert (sql["mean_score_low"], sql["mean_score_high"]) == pytest.approx(
            np.quantile(sql_draws, [0.025, 0.975]),
        )
        for row in panel.loc[list(SERIALIZATION_FORMAT_ORDER)].itertuples():
            assert row.mean_score == pytest.approx(direct_mean)
            assert row.mean_score != pytest.approx(direct_scores.mean())
            assert row.mean_score_effect == pytest.approx(direct_mean - sql_mean)
            assert (row.mean_score_low, row.mean_score_high) == pytest.approx(
                np.quantile(direct_draws, [0.025, 0.975]),
            )
            assert (row.mean_score_effect_low, row.mean_score_effect_high) == pytest.approx(
                np.quantile(direct_draws - sql_draws, [0.025, 0.975]),
            )


def test_tier_weighted_recombination_recovers_existing_overall_scores_and_effects() -> None:
    data = _matrix()
    tiered = calculate_pipeline_size_retrieval_comparison(data, _token_report())
    overall = calculate_pipeline_retrieval_comparison(data)
    columns = ["method", "metric"]
    expected_groups = overall.summary.groupby(columns, observed=True)
    for key, group in tiered.summary.groupby(columns, observed=True):
        expected = expected_groups.get_group(key)
        weights = group["observation_count"]
        assert weights.sum() == expected["observation_count"].sum()
        for value in ("mean_score", "mean_score_effect"):
            assert np.average(group[value], weights=weights) == pytest.approx(
                np.average(expected[value], weights=expected["observation_count"]),
            )


def test_pooled_coverage_has_three_tiers_without_multiplying_pairs_by_methods() -> None:
    analysis = calculate_pipeline_size_retrieval_comparison(_matrix(), _token_report())
    columns = ["database_count", "question_count", "model_count", "observation_count"]
    assert "dataset" not in analysis.coverage
    coverage = analysis.coverage.set_index("size_tier")
    assert set(coverage.index) == set(SIZE_TIER_ORDER)
    assert len(coverage) == len(SIZE_TIER_ORDER)
    expected = {
        "small": [2, 4, 2, 8],
        "medium": [1, 2, 2, 4],
        "large": [3, 6, 2, 12],
    }
    for key, counts in expected.items():
        assert coverage.loc[key, columns].tolist() == counts


def test_empty_pooled_tiers_have_zero_coverage_without_dataset_rows() -> None:
    data = _matrix().loc[lambda frame: frame["dataset"].eq("bird")]
    report = _token_report().loc[lambda frame: frame["dataset"].eq("bird")]
    analysis = calculate_pipeline_size_retrieval_comparison(data, report)
    assert "dataset" not in analysis.summary
    assert "dataset" not in analysis.coverage
    assert len(analysis.coverage) == len(SIZE_TIER_ORDER)
    coverage = analysis.coverage.set_index("size_tier")
    columns = ["database_count", "question_count", "model_count", "observation_count"]
    for tier in ("small", "medium"):
        assert coverage.loc[tier, columns].eq(0).all()
    assert coverage.loc["large", columns].tolist() == [2, 4, 2, 8]
    assert set(analysis.summary["size_tier"]) == {"large"}


def test_same_names_across_datasets_are_distinct_bootstrap_clusters_and_questions() -> None:
    databases = (
        ("spider", "shared", 1_000, 1, 1),
        ("bird", "shared", 1_000, 3, 0),
    )
    data = _matrix(databases)
    analysis = calculate_pipeline_size_retrieval_comparison(data, _token_report(databases))
    assert len(analysis.database_sizes) == 2
    assert len(analysis.records) == len(data)
    assert set(analysis.records["dataset"]) == {"spider", "bird"}
    assert "dataset" not in analysis.summary
    assert "dataset" not in analysis.coverage
    coverage = analysis.coverage.set_index("size_tier").loc["small"]
    assert coverage["database_count"] == 2
    assert coverage["question_count"] == 4  # shared:0 in each dataset counts separately.
    assert coverage["observation_count"] == 8
    assert coverage["model_count"] == 2
    panel = analysis.summary.loc[
        analysis.summary["metric"].eq("normalized_exact_match")
    ].set_index("method")
    assert panel["database_count"].eq(2).all()
    assert panel["question_count"].eq(4).all()
    assert panel["observation_count"].eq(8).all()
    assert panel["model_count"].eq(2).all()
    assert panel.loc["text_to_sql", "mean_score"] == pytest.approx(0.75)
    # Sampling two distinct clusters permits all-success and all-failure draws.
    # Merging the names or stratifying by dataset would yield a degenerate interval.
    for row in panel.loc[list(SERIALIZATION_FORMAT_ORDER)].itertuples():
        assert row.mean_score == pytest.approx(0.25)
        assert row.mean_score_effect == pytest.approx(-0.5)
        assert (row.mean_score_low, row.mean_score_high) == pytest.approx((0, 1))
        assert (row.mean_score_effect_low, row.mean_score_effect_high) == pytest.approx((-1, 1))


def test_missing_serialization_database_cannot_silently_remove_runs() -> None:
    report = _token_report()
    # BIRD still has a database named shared, so the join must include dataset.
    report = report.loc[~(report["dataset"].eq("spider") & report["database_id"].eq("shared"))]
    with pytest.raises(ValueError):
        calculate_pipeline_size_retrieval_comparison(_matrix(), report)


@pytest.mark.parametrize("problem", [
    "missing_direct", "missing_sql", "duplicate_direct", "duplicate_sql",
    "wrong_model", "wrong_database", "missing_format", "null_identity",
])
def test_unequal_or_duplicate_question_model_pairs_are_rejected(problem: str) -> None:
    data = _matrix()
    direct = data.index[data["pipeline"].eq("direct")][0]
    sql = data.index[data["pipeline"].eq("text_to_sql")][0]
    if problem.startswith("missing_") and problem != "missing_format":
        data = data.drop(sql if problem == "missing_sql" else direct)
    elif problem.startswith("duplicate_"):
        index = sql if problem == "duplicate_sql" else direct
        data = pd.concat([data, data.loc[[index]]], ignore_index=True)
    elif problem == "wrong_model":
        data.loc[direct, "model_id"] = "different"
    elif problem == "wrong_database":
        data.loc[direct, "db_id"] = "small_many"
    elif problem == "missing_format":
        data = data.loc[data["format"].ne(SERIALIZATION_FORMAT_ORDER[0])]
    else:
        data.loc[direct, "example_id"] = None
    with pytest.raises(ValueError):
        calculate_pipeline_size_retrieval_comparison(data, _token_report())


@pytest.mark.parametrize("failure", ["context", "sql_execution", "sql_extraction"])
def test_failures_require_zero_scores_and_stay_in_the_tier_denominator(failure: str) -> None:
    data = _matrix()
    pipeline = "direct" if failure == "context" else "text_to_sql"
    index = data.index[data["pipeline"].eq(pipeline)][0]
    data.loc[index, SCORE_COLUMNS] = 1
    data.loc[index, "has_answer"] = False
    if failure == "context":
        data.loc[index, "error"] = "context_exceeded"
    elif failure == "sql_execution":
        data.loc[index, "execution_success"] = False
    else:
        data.loc[index, "error"] = "no_sql_extracted"
    with pytest.raises(ValueError, match="Failed outputs must have zero"):
        calculate_pipeline_size_retrieval_comparison(data, _token_report())
    data.loc[index, SCORE_COLUMNS] = 0
    analysis = calculate_pipeline_size_retrieval_comparison(data, _token_report())
    assert len(analysis.records) == len(data)
    panel = analysis.summary.loc[
        analysis.summary["size_tier"].eq("small")
    ]
    assert panel["observation_count"].eq(8).all()
    method = "text_to_sql" if pipeline == "text_to_sql" else data.loc[index, "format"]
    for metric in DIRECT_ACCURACY_METRIC_ORDER:
        expected = data.loc[
            data["dataset"].eq("spider") & data["db_id"].isin(["shared", "small_many"])
            & (data["pipeline"].eq("text_to_sql") if pipeline == "text_to_sql"
               else data["format"].eq(method)), f"scores.{metric}",
        ].mean()
        actual = panel.loc[panel["method"].eq(method) & panel["metric"].eq(metric), "mean_score"]
        assert actual.item() == pytest.approx(expected)


def test_output_answer_flags_do_not_reselect_the_callers_cohort() -> None:
    data = _matrix()
    data["has_answer"] = False
    data.loc[0, "scores.bertscore_f"] = -0.1
    analysis = calculate_pipeline_size_retrieval_comparison(data, _token_report())
    assert len(analysis.records) == len(data)
    assert analysis.records["scores.bertscore_f"].min() == -0.1


def test_excluded_models_are_removed_before_pairing_score_and_database_validation() -> None:
    data = _matrix()
    excluded = data.iloc[[0]].copy()
    excluded["model_id"] = "Qwen3.5-27B-Q4"
    excluded["db_id"] = "absent_from_token_report"
    excluded["scores.bertscore_f"] = np.nan
    data = pd.concat([data, excluded], ignore_index=True)
    original = data.copy(deep=True)
    analysis = calculate_pipeline_size_retrieval_comparison(data, _token_report())
    assert len(analysis.records) == len(data) - 1
    assert not analysis.records["model_id"].str.contains("27B").any()
    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(ValueError):
        calculate_pipeline_size_retrieval_comparison(
            data, _token_report(), excluded_model_markers=(),
        )


def test_single_database_tiers_have_no_spurious_confidence_intervals() -> None:
    analysis = calculate_pipeline_size_retrieval_comparison(_matrix(), _token_report())
    singleton = analysis.summary.loc[analysis.summary["database_count"].eq(1)]
    assert not singleton.empty
    interval_columns = [
        "mean_score_low", "mean_score_high", "mean_score_effect_low", "mean_score_effect_high",
    ]
    assert singleton[interval_columns].isna().all(axis=None)
    assert singleton[["mean_score", "mean_score_effect"]].notna().all(axis=None)
    multiple = analysis.summary.loc[analysis.summary["database_count"].gt(1)]
    assert multiple[interval_columns].notna().all(axis=None)
