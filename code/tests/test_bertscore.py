"""Hermetic unit tests for the lazy BERTScore wrapper."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from metrics import bertscore


class _Tensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


@pytest.fixture(autouse=True)
def clear_scorer_cache() -> Any:
    bertscore._scorer.cache_clear()
    yield
    bertscore._scorer.cache_clear()


def _install_fake_bert_score(
    monkeypatch: pytest.MonkeyPatch,
    *,
    score_values: tuple[list[float], list[float], list[float]] | None = None,
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    outputs = score_values or ([0.81], [0.72], [0.76])

    class FakeBERTScorer:
        def __init__(self, **kwargs: Any) -> None:
            self.config = kwargs
            self.calls: list[tuple[list[str], list[str]]] = []
            instances.append({"scorer": self, "config": kwargs})

        def score(self, predictions: list[str], references: list[str]) -> tuple[_Tensor, ...]:
            self.calls.append((predictions, references))
            return tuple(_Tensor(values) for values in outputs)

    fake_module = ModuleType("bert_score")
    fake_module.BERTScorer = FakeBERTScorer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bert_score", fake_module)
    return instances


def test_configures_raw_bertscore_and_reuses_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_bert_score(monkeypatch)

    first = bertscore.bert_score_batch(
        ["prediction"], ["reference"], model_type="fake-model", lang="fr"
    )
    second = bertscore.bert_score_batch(
        ["another"], ["reference 2"], model_type="fake-model", lang="fr"
    )

    assert first == [(0.81, 0.72, 0.76)]
    assert second == [(0.81, 0.72, 0.76)]
    assert len(instances) == 1
    assert instances[0]["config"] == {
        "model_type": "fake-model",
        "lang": "fr",
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
    }
    assert instances[0]["scorer"].calls == [
        (["prediction"], ["reference"]),
        (["another"], ["reference 2"]),
    ]


def test_returns_raw_precision_recall_f1_without_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_bert_score(
        monkeypatch,
        score_values=([1.2], [-0.1], [1.4]),
    )

    assert bertscore.bert_score_batch(["p"], ["r"]) == [(1.2, -0.1, 1.4)]


def test_empty_text_is_replaced_before_scorer_call(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = _install_fake_bert_score(monkeypatch)

    bertscore.bert_score_batch(["", "  ", "text"], [" ", "reference", ""])

    assert instances[0]["scorer"].calls == [([".", ".", "text"], [".", "reference", "."])]


def test_misaligned_inputs_fail_and_empty_batch_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_fake_bert_score(monkeypatch)

    assert bertscore.bert_score_batch([], []) == []
    assert instances == []
    with pytest.raises(ValueError, match="must align"):
        bertscore.bert_score_batch(["prediction"], [])
    assert instances == []
