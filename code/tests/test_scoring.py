"""Tests for scoring orchestration (metrics.scoring), using injected fake metric fns."""

from __future__ import annotations

from datasets.types import DatasetExample
from metrics.bertscore import ScoreTriple
from metrics.scoring import (
    has_order_by,
    prediction_text,
    reference_text,
    references_from_rows,
    rows_to_text,
    score_records,
    scoring_text_pair,
)
from pipelines.shared import base_result


def _ex(idx: int, db: str = "db") -> DatasetExample:
    return DatasetExample(
        dataset="spider",
        db_id=db,
        question="q",
        gold_sql="SELECT name FROM t",
        evidence=None,
        difficulty=None,
        example_id=f"spider:{db}:{idx}",
    )


def _fake_exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred == ref else 0.0


def _fake_rouge(pred: str, ref: str) -> float:
    return 1.0 if pred == ref else 0.0


def _fake_bert(preds: list[str], refs: list[str]) -> list[ScoreTriple]:
    return [(0.9, 0.8, 0.85) for _ in preds]


def _ref(rows: list[dict] | None, gold_sql: str = "SELECT name FROM t") -> dict:
    return {"example_id": "spider:db:0", "rows": rows, "gold_sql": gold_sql}


def test_rows_to_text() -> None:
    assert rows_to_text([{"a": 1, "b": "x"}, {"a": 2, "b": None}]) == "1 | x\n2 | "
    assert rows_to_text([]) == ""
    assert rows_to_text(None) == ""


def test_prediction_text_per_pipeline() -> None:
    direct = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    direct["raw_output"] = "five singers"
    assert prediction_text(direct) == "five singers"

    sql = base_result(_ex(0), pipeline="text_to_sql", fmt=None, model_id="m")
    sql["predicted_rows"] = [{"c": 5}]
    assert prediction_text(sql) == "5"


def test_reference_text() -> None:
    refs = references_from_rows([_ref([{"n": "Alice"}], gold_sql="x")])
    assert reference_text(refs["spider:db:0"]) == "Alice"


def test_direct_empty_reference_uses_prompt_no_answer_sentinel() -> None:
    direct = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    direct["raw_output"] = "N/A"
    refs = references_from_rows([_ref([])])

    assert scoring_text_pair(direct, refs["spider:db:0"]) == ("N/A", "N/A")


def test_text_to_sql_empty_reference_remains_empty() -> None:
    sql = base_result(_ex(0), pipeline="text_to_sql", fmt=None, model_id="m")
    sql["predicted_rows"] = []
    refs = references_from_rows([_ref([])])

    assert scoring_text_pair(sql, refs["spider:db:0"]) == ("", "")


def test_has_order_by() -> None:
    assert has_order_by("SELECT x FROM t ORDER BY x")
    assert not has_order_by("SELECT x FROM t")
    # ORDER BY inside a subquery does not order the outer result
    assert not has_order_by("SELECT x FROM (SELECT x FROM t ORDER BY x) WHERE x > 1")
    assert has_order_by("SELECT x FROM (SELECT x FROM t) ORDER BY x")


def test_score_records_direct_pipeline() -> None:
    r = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    r["raw_output"] = "Alice"
    refs = references_from_rows([_ref([{"name": "Alice"}])])
    scored, stats = score_records(
        [r],
        refs,
        exact_match_fn=_fake_exact_match,
        rouge_fn=_fake_rouge,
        bert_batch_fn=_fake_bert,
    )
    assert stats["scored"] == 1
    assert scored[0]["scores"]["normalized_exact_match"] == 1.0
    assert scored[0]["scores"]["rouge_l_f"] == 1.0  # "Alice" == "Alice"
    assert scored[0]["scores"]["bertscore_f"] == 0.85
    assert scored[0]["execution"] is None  # not text_to_sql


def test_score_records_direct_no_answer_matches_empty_gold_result() -> None:
    r = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    r["raw_output"] = "N/A"
    refs = references_from_rows([_ref([])])

    captured: dict[str, list[str]] = {}

    def capture_bert(preds: list[str], reference_texts: list[str]) -> list[ScoreTriple]:
        captured["predictions"] = preds
        captured["references"] = reference_texts
        return _fake_bert(preds, reference_texts)

    scored, _ = score_records(
        [r],
        refs,
        exact_match_fn=_fake_exact_match,
        rouge_fn=_fake_rouge,
        bert_batch_fn=capture_bert,
    )

    assert scored[0]["scores"]["normalized_exact_match"] == 1.0
    assert scored[0]["scores"]["rouge_l_f"] == 1.0
    assert captured == {"predictions": ["N/A"], "references": ["N/A"]}


def test_score_records_text_to_sql_execution() -> None:
    r = base_result(_ex(0), pipeline="text_to_sql", fmt=None, model_id="m")
    r["predicted_rows"] = [{"name": "Bob"}]
    r["execution_success"] = True
    refs = references_from_rows([_ref([{"name": "Alice"}])])
    scored, stats = score_records(
        [r],
        refs,
        exact_match_fn=_fake_exact_match,
        rouge_fn=_fake_rouge,
        bert_batch_fn=_fake_bert,
    )
    assert scored[0]["scores"]["normalized_exact_match"] == 0.0
    assert scored[0]["execution"]["execution_success"] is True
    assert scored[0]["execution"]["result_match"] is False
    assert scored[0]["execution"]["execution_accuracy"] == 1.0
    assert stats["execution_scored"] == 1
    assert scored[0]["scores"] is not None  # text-scored too


def test_score_records_zeroes_failed_output_instead_of_matching_empty_reference() -> None:
    r = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    r["error"] = "context_exceeded"
    refs = references_from_rows([_ref([])])

    def metric_must_not_run(pred: str, ref: str) -> float:
        raise AssertionError(f"failed output was scored: {pred!r} vs {ref!r}")

    def bert_must_not_run(preds: list[str], refs: list[str]) -> list[ScoreTriple]:
        raise AssertionError(f"failed output was BERT-scored: {preds!r} vs {refs!r}")

    scored, stats = score_records(
        [r],
        refs,
        exact_match_fn=metric_must_not_run,
        rouge_fn=metric_must_not_run,
        bert_batch_fn=bert_must_not_run,
    )

    assert scored[0]["scores"] == {
        "normalized_exact_match": 0.0,
        "rouge_l_f": 0.0,
        "bertscore_p": 0.0,
        "bertscore_r": 0.0,
        "bertscore_f": 0.0,
    }
    assert stats["scored"] == 1
    assert stats["failed_output_zeroed"] == 1
    assert stats["skipped_no_reference"] == 0


def test_score_records_zeroes_definitive_text_to_sql_failure() -> None:
    r = base_result(_ex(0), pipeline="text_to_sql", fmt=None, model_id="m")
    r["execution_success"] = False
    r["definitive_failure"] = True
    refs = references_from_rows([_ref([])])

    scored, stats = score_records([r], refs, with_bertscore=False)

    assert scored[0]["scores"]["normalized_exact_match"] == 0.0
    assert scored[0]["scores"]["rouge_l_f"] == 0.0
    assert scored[0]["execution"]["execution_accuracy"] == 0.0
    assert stats["failed_output_zeroed"] == 1


def test_score_records_failed_output_keeps_bertscore_null_when_disabled() -> None:
    r = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    r["error"] = "generation failed"
    refs = references_from_rows([_ref([])])

    scored, stats = score_records([r], refs, with_bertscore=False)

    assert scored[0]["scores"]["normalized_exact_match"] == 0.0
    assert scored[0]["scores"]["rouge_l_f"] == 0.0
    assert scored[0]["scores"]["bertscore_p"] is None
    assert scored[0]["scores"]["bertscore_r"] is None
    assert scored[0]["scores"]["bertscore_f"] is None
    assert stats["failed_output_zeroed"] == 1


def test_score_records_skips_missing_reference() -> None:
    r = base_result(_ex(99), pipeline="direct", fmt="json", model_id="m")
    r["raw_output"] = "x"
    scored, stats = score_records([r], {}, rouge_fn=_fake_rouge, bert_batch_fn=_fake_bert)
    assert stats["scored"] == 0
    assert stats["skipped_no_reference"] == 1
    assert scored[0]["scores"] is None


def test_score_records_skips_reference_with_gold_error() -> None:
    r = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    refs = references_from_rows([{"example_id": "spider:db:0", "rows": None, "gold_sql": "bad"}])
    scored, stats = score_records([r], refs, rouge_fn=_fake_rouge, bert_batch_fn=_fake_bert)
    assert stats["scored"] == 0
    assert scored[0]["scores"] is None


def test_score_records_no_bertscore() -> None:
    r = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    r["raw_output"] = "Alice"
    refs = references_from_rows(
        [{"example_id": "spider:db:0", "rows": [{"name": "Alice"}], "gold_sql": "x"}]
    )

    # bert_batch_fn must NOT be called when disabled
    def _boom(preds, refs):  # noqa: ANN001
        raise AssertionError("bertscore should be skipped")

    scored, stats = score_records(
        [r],
        refs,
        with_bertscore=False,
        exact_match_fn=_fake_exact_match,
        rouge_fn=_fake_rouge,
        bert_batch_fn=_boom,
    )
    assert stats["bertscore_enabled"] is False
    # None (not 0.0) so "not computed" stays distinguishable from "scored zero"
    assert scored[0]["scores"]["bertscore_f"] is None
    assert scored[0]["scores"]["rouge_l_f"] == 1.0


def test_score_records_force_order_sensitive() -> None:
    r = base_result(_ex(0), pipeline="text_to_sql", fmt=None, model_id="m")
    r["predicted_rows"] = [{"x": 2}, {"x": 1}]
    refs = references_from_rows(
        [{"example_id": "spider:db:0", "rows": [{"x": 1}, {"x": 2}], "gold_sql": "SELECT x FROM t"}]
    )
    scored, _ = score_records(
        [r],
        refs,
        force_order_sensitive=True,
        exact_match_fn=_fake_exact_match,
        rouge_fn=_fake_rouge,
        bert_batch_fn=_fake_bert,
    )
    assert scored[0]["execution"]["result_match"] is False
    assert scored[0]["execution"]["execution_accuracy"] == 1.0


def test_score_records_does_not_mutate_input() -> None:
    r = base_result(_ex(0), pipeline="direct", fmt="json", model_id="m")
    r["raw_output"] = "Alice"
    refs = references_from_rows(
        [{"example_id": "spider:db:0", "rows": [{"name": "Alice"}], "gold_sql": "x"}]
    )
    score_records(
        [r],
        refs,
        exact_match_fn=_fake_exact_match,
        rouge_fn=_fake_rouge,
        bert_batch_fn=_fake_bert,
    )
    assert r["scores"] is None  # original untouched (scored copies returned)
