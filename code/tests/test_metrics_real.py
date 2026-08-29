"""Tests against the real metric implementations (needs the ``[metrics]`` extra).

BERTScore's model is never loaded here — only the pure input-validation paths of
``bert_score_batch`` are exercised.
"""

from __future__ import annotations

import pytest

from metrics.bertscore import bert_score_batch

pytest.importorskip("rouge_score", reason="requires the [metrics] extra")
pytest.importorskip("transformers", reason="requires the [metrics] extra")

from metrics.exact_match import normalized_exact_match  # noqa: E402
from metrics.rouge import rouge_l  # noqa: E402


def test_normalized_exact_match_accepts_formatting_mutations() -> None:
    assert normalized_exact_match("NEW YORK,\nUSA", "new york usa") == 1.0
    assert normalized_exact_match("1,000", "1000") == 1.0


def test_normalized_exact_match_rejects_different_results() -> None:
    assert normalized_exact_match("Alice and Bob", "Alice and Carol") == 0.0
    assert normalized_exact_match("Alice Bob", "Bob Alice") == 0.0


def test_rouge_l_identical_texts_score_one() -> None:
    assert rouge_l("five singers", "five singers") == pytest.approx(1.0)


def test_rouge_l_disjoint_texts_score_zero() -> None:
    assert rouge_l("apples oranges", "quantum flux") == 0.0


def test_rouge_l_partial_overlap_is_between() -> None:
    score = rouge_l("there are five singers", "five singers")
    assert 0.0 < score < 1.0


def test_bert_score_batch_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        bert_score_batch(["a", "b"], ["a"])


def test_bert_score_batch_empty_input_short_circuits() -> None:
    # Must return [] without loading the torch model.
    assert bert_score_batch([], []) == []
