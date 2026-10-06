"""Statistical invariants: pairing, weighting, and the independent sampling unit."""

import numpy as np
import pandas as pd
import pytest

from visualizations.uncertainty import mean_intervals


def _paired_data():
    return pd.DataFrame([
        {"dataset": "spider", "db_id": db, "example_id": f"{db}:{question}",
         "format": fmt, "value": value + offset}
        for db, count, value in (("small", 1, 1.0), ("large", 9, 10.0))
        for question in range(count)
        for fmt, offset in (("csv", 0.0), ("json", 2.0))
    ])


def _summarize(data, effect="difference"):
    return mean_intervals(
        data, ["dataset"], {"value": "value"}, comparison="format", reference="csv",
        effect=effect, pair_keys=("db_id", "example_id"),
    ).set_index("format")


def test_paired_constant_gain_has_zero_uncertainty_despite_variable_database_means():
    result = _summarize(_paired_data())
    assert result.loc["csv", "value_low"] == 1
    assert result.loc["csv", "value_high"] == 10
    assert result.loc["json", "value_effect"] == pytest.approx(2)
    assert result.loc["json", "value_effect_low"] == pytest.approx(2)
    assert result.loc["json", "value_effect_high"] == pytest.approx(2)


def test_ratio_uses_question_weighted_means_and_not_mean_of_question_ratios():
    result = _summarize(_paired_data(), effect="ratio")
    assert result.loc["json", "value_effect"] == pytest.approx(11.1 / 9.1)
    assert result.loc["json", "value_effect_low"] == pytest.approx(1.2)
    assert result.loc["json", "value_effect_high"] == pytest.approx(3.0)


def test_repeating_questions_does_not_inflate_independent_sample_size():
    original = _paired_data()
    duplicated = pd.concat([original, original.assign(example_id=original.example_id + ":copy")])
    pd.testing.assert_frame_equal(_summarize(original), _summarize(duplicated))


def test_missing_pair_is_rejected_and_single_database_interval_is_not_estimable():
    with pytest.raises(ValueError, match="unequal observation identities"):
        _summarize(_paired_data().iloc[:-1])
    result = _summarize(_paired_data().query("db_id == 'small'"))
    assert np.isnan(result.loc["json", "value_effect_low"])
    assert result.loc["json", "value_effect"] == 2


def test_zero_reference_ratio_is_not_reported_as_finite():
    data = _paired_data()
    data.loc[data["format"].eq("csv"), "value"] = 0
    result = _summarize(data, effect="ratio")
    assert np.isnan(result.loc["json", "value_effect"])
    assert np.isnan(result.loc["json", "value_effect_low"])
