"""Regression coverage for independent, fixed-roster Chapter 5 question cohorts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from visualizations.cohorts import (
    COHORT_COLUMNS,
    DIRECT_FORMATS,
    RETAINED_MODELS,
    build_analysis_cohorts,
    generate_cohort_artifacts,
)


def _records(example_id: str = "question") -> pd.DataFrame:
    rows = []
    for model in RETAINED_MODELS:
        for pipeline, fmt in [*(("direct", fmt) for fmt in DIRECT_FORMATS), ("text_to_sql", None)]:
            sql = pipeline == "text_to_sql"
            rows.append({
                "dataset": "spider", "example_id": example_id, "db_id": "database",
                "model_id": model, "pipeline": pipeline, "format": fmt,
                "error": None, "input_tokens": 100, "retry_count": 0,
                "attempts.attempt_index": [0] if sql else None,
                "attempts.error": [None] if sql else None,
                "attempts.llm_metrics.error": [None] if sql else None,
                "attempts.llm_metrics.input_tokens": [100] if sql else None,
                "has_answer": True, "execution_success": True if sql else None,
                "scores.normalized_exact_match": 0.5,
                "generation_time_s": 1.0,
            })
    return pd.DataFrame(rows)


def _references(tmp_path: Path, records: list[dict] | None = None, **overrides) -> Path:
    reference = {
        "example_id": "question", "db_id": "database", "rows": [{"value": 1}],
        "row_count": 1, "error": None,
    }
    reference.update(overrides)
    (tmp_path / "spider_references.ndjson").write_text(
        "\n".join(json.dumps(row) for row in (records or [reference])) + "\n", encoding="utf-8",
    )
    (tmp_path / "bird_references.ndjson").write_text("", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("value", [0, None])
def test_zero_and_null_reference_rows_are_nonempty(tmp_path, value):
    refs = _references(tmp_path, records=[{
        "example_id": "question", "db_id": "database", "rows": [{"value": value}],
        "row_count": 1, "error": None,
    }])
    cohorts = build_analysis_cohorts(_records(), refs)
    assert cohorts.manifest.iloc[0]["answerable"]
    assert len(cohorts.answerable_records) == 70
    assert len(cohorts.context_fit_records) == 70


def test_empty_references_are_excluded_only_from_answerable(tmp_path):
    refs = _references(tmp_path, records=[{
        "example_id": "question", "db_id": "database", "rows": [],
        "row_count": 0, "error": None,
    }])
    cohorts = build_analysis_cohorts(_records(), refs)
    assert cohorts.answerable_records.empty
    assert len(cohorts.context_fit_records) == 70
    assert cohorts.summary.iloc[0]["context_fit_empty_reference_questions"] == 1
    assert cohorts.summary.iloc[0]["intersection_questions"] == 0


def test_context_failure_is_kept_with_zero_accuracy(tmp_path):
    data = _records()
    data.loc[0, ["error", "input_tokens", "scores.normalized_exact_match"]] = [
        "context_exceeded", 0, 0,
    ]
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert len(cohorts.answerable_records) == 70
    assert cohorts.answerable_records.loc[0, "scores.normalized_exact_match"] == 0
    assert cohorts.context_fit_records.empty
    assert cohorts.manifest.iloc[0]["context_status"] == "not_fit"


def test_failed_sql_blank_answers_and_all_retry_work_are_retained(tmp_path):
    data = _records()
    data["has_answer"] = False
    sql = data["pipeline"].eq("text_to_sql")
    data.loc[sql, "execution_success"] = False
    for index in data.index[sql]:
        data.at[index, "retry_count"] = 3
        data.at[index, "input_tokens"] = 460
        data.at[index, "attempts.attempt_index"] = [0, 1, 2, 3]
        data.at[index, "attempts.error"] = ["no SQL statement found in the response"] * 4
        data.at[index, "attempts.llm_metrics.error"] = [None] * 4
        data.at[index, "attempts.llm_metrics.input_tokens"] = [100, 110, 120, 130]
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert len(cohorts.context_fit_records) == 70
    assert cohorts.context_fit_records.loc[sql, "input_tokens"].sum() == 4600
    methods = cohorts.context_fit_records.copy()
    methods["method"] = methods["format"].fillna("text_to_sql")
    key_columns = ["dataset", "db_id", "example_id", "model_id"]
    identities = [
        set(group[key_columns].itertuples(index=False, name=None))
        for _, group in methods.groupby("method")
    ]
    assert len(identities) == 7
    assert all(keys == identities[0] for keys in identities)


@pytest.mark.parametrize("error", [
    "context_exceeded", "Requested tokens (200000) exceed context window of 131072",
    "This model's maximum context length is 131072 tokens",
])
def test_sql_correction_context_errors_exclude_question(tmp_path, error):
    data = _records()
    index = data.index[data["pipeline"].eq("text_to_sql")][0]
    data.at[index, "retry_count"] = 1
    data.at[index, "attempts.attempt_index"] = [0, 1]
    data.at[index, "attempts.error"] = ["syntax error", error]
    data.at[index, "attempts.llm_metrics.error"] = [None, error]
    data.at[index, "attempts.llm_metrics.input_tokens"] = [100, 0]
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert cohorts.context_fit_records.empty
    assert cohorts.manifest.iloc[0]["context_status"] == "not_fit"
    assert len(cohorts.answerable_records) == 70


@pytest.mark.parametrize("backend_error,expected_status", [
    (None, "fit"),
    ("Requested tokens (200000) exceed context window of 131072", "not_fit"),
])
def test_sql_identifiers_cannot_be_mistaken_for_backend_context_errors(
    tmp_path, backend_error, expected_status,
):
    data = _records()
    index = data.index[data["pipeline"].eq("text_to_sql")][0]
    data.at[index, "execution_success"] = False
    data.at[index, "attempts.error"] = ["no such column: context_length"]
    data.at[index, "attempts.llm_metrics.error"] = [backend_error]
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert cohorts.manifest.iloc[0]["context_status"] == expected_status
    assert len(cohorts.context_fit_records) == (70 if backend_error is None else 0)
    assert len(cohorts.answerable_records) == 70


@pytest.mark.parametrize("drop", ["record", "model", "format", "question"])
def test_missing_expected_matrix_is_unknown_not_a_smaller_roster(tmp_path, drop):
    data = _records()
    if drop == "record":
        data = data.iloc[1:]
    elif drop == "model":
        data = data.loc[data["model_id"].ne(RETAINED_MODELS[0])]
    elif drop == "format":
        data = data.loc[data["format"].ne("csv")]
    else:
        data = data.iloc[0:0]
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert cohorts.context_fit_records.empty
    assert cohorts.manifest.iloc[0]["context_status"] == "unknown"
    assert cohorts.manifest.iloc[0]["missing_records"] > 0
    assert cohorts.summary.iloc[0]["incomplete_questions"] == 1


@pytest.mark.parametrize("field,value", [
    ("error", "connection reset"), ("input_tokens", 0),
])
def test_unrelated_direct_failure_does_not_establish_feasibility(tmp_path, field, value):
    data = _records()
    data.at[0, field] = value
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert cohorts.manifest.iloc[0]["context_status"] == "unknown"
    assert cohorts.context_fit_records.empty
    assert len(cohorts.answerable_records) == 70


@pytest.mark.parametrize("field,value", [
    ("attempts.attempt_index", None), ("attempts.attempt_index", [1]),
    ("attempts.llm_metrics.error", ["connection reset"]),
    ("attempts.llm_metrics.input_tokens", [0]),
    ("attempts.llm_metrics.input_tokens", [99]), ("retry_count", 1),
])
def test_unknown_or_inconsistent_sql_attempts_do_not_masquerade_as_fit(tmp_path, field, value):
    data = _records()
    index = data.index[data["pipeline"].eq("text_to_sql")][0]
    data.at[index, field] = value
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert cohorts.manifest.iloc[0]["context_status"] == "unknown"
    assert cohorts.context_fit_records.empty


@pytest.mark.parametrize("kind", ["duplicate", "model", "format", "database", "reference_database"])
def test_identity_ambiguities_raise(tmp_path, kind):
    data = _records()
    refs = _references(tmp_path, db_id="other" if kind == "reference_database" else "database")
    if kind == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif kind in ("model", "format", "database"):
        field = {"model": "model_id", "format": "format", "database": "db_id"}[kind]
        data.at[0, field] = "unexpected"
    with pytest.raises(ValueError, match="Duplicate|Unexpected|identity"):
        build_analysis_cohorts(data, refs)


@pytest.mark.parametrize("status,overrides", [
    ("sql_error", {"error": "syntax error", "rows": None, "row_count": None}),
    ("inconsistent_rows", {"row_count": 0}),
    ("inconsistent_rows", {"rows": None}),
])
def test_reference_errors_do_not_control_context(tmp_path, status, overrides):
    cohorts = build_analysis_cohorts(_records(), _references(tmp_path, **overrides))
    assert cohorts.manifest.iloc[0]["reference_status"] == status
    assert cohorts.manifest.iloc[0]["answerable_status"] == "unknown"
    assert cohorts.summary.iloc[0]["empty_references"] == 0
    assert cohorts.answerable_records.empty
    assert len(cohorts.context_fit_records) == 70


def test_missing_reference_is_unknown_and_duplicate_reference_raises(tmp_path):
    cohorts = build_analysis_cohorts(_records(), tmp_path)
    assert cohorts.manifest.iloc[0]["reference_status"] == "missing"
    assert len(cohorts.context_fit_records) == 70
    refs = _references(tmp_path)
    path = refs / "spider_references.ndjson"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate reference"):
        build_analysis_cohorts(_records(), refs)


def test_missing_load_columns_raise(tmp_path):
    with pytest.raises(ValueError, match="Missing cohort load columns"):
        build_analysis_cohorts(_records().drop(columns=COHORT_COLUMNS[-1]), tmp_path)


def test_membership_does_not_hide_invalid_timing_measurements(tmp_path):
    data = _records()
    data.loc[0, "generation_time_s"] = float("nan")
    cohorts = build_analysis_cohorts(data, _references(tmp_path))
    assert len(cohorts.context_fit_records) == 70
    assert pd.isna(cohorts.context_fit_records.loc[0, "generation_time_s"])


def test_manifest_and_counts_are_reproducible_under_input_permutation(tmp_path):
    refs = _references(tmp_path)
    one = build_analysis_cohorts(_records(), refs)
    two = build_analysis_cohorts(_records().sample(frac=1, random_state=42), refs)
    pd.testing.assert_frame_equal(one.manifest, two.manifest)
    pd.testing.assert_frame_equal(one.summary, two.summary)
    assert one.audit == two.audit
    files = generate_cohort_artifacts(one, tmp_path / "tables", tmp_path / "summaries")
    before = [path.read_bytes() for path in files]
    generate_cohort_artifacts(two, tmp_path / "tables", tmp_path / "summaries")
    assert before == [path.read_bytes() for path in files]
    assert json.loads(files[-1].read_text())["datasets"][0]["context_fit_questions"] == 1
    assert r"\parbox" not in files[0].read_text()
    assert "Answerable:" not in files[0].read_text()
    assert r"\resizebox{\linewidth}{!}" in files[0].read_text()
