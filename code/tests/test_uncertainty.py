"""Statistical invariants: pairing, weighting, and the independent sampling unit."""

import numpy as np
import pandas as pd
import pytest

from visualizations.uncertainty import interval_marks, mean_intervals, table_estimate


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


@pytest.mark.parametrize("effect", ["difference", "ratio"])
def test_context_fit_threshold_omits_two_cluster_intervals_but_keeps_point_effects(effect):
    data = _paired_data()
    original = _summarize(data, effect=effect)
    result = mean_intervals(
        data, ["dataset"], {"value": "value"}, comparison="format", reference="csv",
        effect=effect, pair_keys=("db_id", "example_id"), min_databases=3,
    ).set_index("format")
    assert result.filter(regex="_(low|high)$").isna().all(axis=None)
    pd.testing.assert_series_equal(result["value_effect"], original["value_effect"])
    assert original.filter(regex="_(low|high)$").notna().all(axis=None)


def test_intentionally_omitted_table_interval_is_a_plain_point_estimate():
    row = {"value": 2.0, "value_low": np.nan, "value_high": np.nan}
    assert table_estimate(row, "value", show_interval=False) == r"\num{2.00}"
    assert "not estimable" in table_estimate(row, "value")


def test_interval_marks_skip_missing_endpoints_without_drawing_empty_artists():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    rows = pd.DataFrame({"value_low": [np.nan, 1.0], "value_high": [np.nan, 3.0]})
    interval_marks(axis, [0, 1], rows, "value", horizontal=True)
    assert len(axis.collections) == 1
    np.testing.assert_allclose(axis.collections[0].get_segments()[0], [[1, 1], [3, 1]])
    plt.close(figure)
