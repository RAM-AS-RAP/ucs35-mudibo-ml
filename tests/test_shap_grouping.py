import inspect

import numpy as np
import pytest

from agrisence.explainability import shap_explainer
from agrisence.explainability.shap_explainer import (
    BASELINE_GROUPS,
    DRIVER_GROUPS,
    TYPICAL_CHANGE_LABEL,
    shap_to_farmer_explanation,
)


DRIVER_FEATURES = [
    "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
    "log_ratio_3m", "log_ratio_12m", "month_sin", "month_cos",
]


def _shap_result(include_year=True, values=None):
    feature_names = [*DRIVER_FEATURES, "market_Nairobi", "commodity_Beans"]
    if include_year:
        feature_names.insert(7, "year")
    if values is None:
        values = np.random.default_rng(42).normal(0.0, 0.01, (3, len(feature_names)))
    return {
        "shap_values": np.asarray(values, dtype=float),
        "base_value": 0.025,
        "feature_names": feature_names,
    }


@pytest.mark.parametrize("include_year", [True, False])
def test_random_shap_rows_reconcile_with_and_without_year(include_year):
    result = _shap_result(include_year=include_year)
    predictions = result["base_value"] + result["shap_values"].sum(axis=1)

    for row_index, prediction in enumerate(predictions):
        explanation = shap_to_farmer_explanation(
            result,
            row_index=row_index,
            origin_price=100.0,
            model_prediction=float(prediction),
        )
        assert explanation["predicted_log_change"] == pytest.approx(prediction, abs=1e-4)


def test_unknown_feature_name_is_rejected():
    result = _shap_result()
    result["feature_names"].append("unexpected_feature")
    result["shap_values"] = np.pad(result["shap_values"], ((0, 0), (0, 1)))

    with pytest.raises(ValueError, match="unexpected_feature"):
        shap_to_farmer_explanation(result, row_index=0, origin_price=100.0)


def test_feature_matching_two_groups_is_rejected(monkeypatch):
    overlapping_baselines = dict(BASELINE_GROUPS)
    overlapping_baselines["Market and crop"] = "PREFIX:market_,commodity_,log_return_lag_1"
    monkeypatch.setattr(shap_explainer, "BASELINE_GROUPS", overlapping_baselines)

    with pytest.raises(ValueError, match="log_return_lag_1"):
        shap_to_farmer_explanation(_shap_result(), row_index=0, origin_price=100.0)


def test_baseline_groups_fold_into_typical_change_not_driver_groups():
    feature_names = [*DRIVER_FEATURES, "year", "market_Nairobi", "commodity_Beans"]
    row_values = np.array([0.0] * len(DRIVER_FEATURES) + [0.04, 0.03, -0.01])
    result = {
        "shap_values": row_values.reshape(1, -1),
        "base_value": 0.02,
        "feature_names": feature_names,
    }

    explanation = shap_to_farmer_explanation(result, row_index=0, origin_price=100.0)

    assert explanation["typical_change"] == {
        "label": TYPICAL_CHANGE_LABEL,
        "log_value": 0.08,
        "pct_value": round((np.exp(0.08) - 1) * 100, 2),
    }
    assert {group["label"] for group in explanation["groups"]} == set(DRIVER_GROUPS)
    assert not set(BASELINE_GROUPS).intersection(group["label"] for group in explanation["groups"])


def test_wrong_model_prediction_raises_assertion():
    result = _shap_result()

    with pytest.raises(AssertionError, match="model_prediction"):
        shap_to_farmer_explanation(
            result, row_index=0, origin_price=100.0, model_prediction=999.0,
        )


def test_all_driver_groups_are_returned_without_top_n_behavior():
    result = _shap_result(include_year=False)

    explanation = shap_to_farmer_explanation(result, row_index=0, origin_price=100.0)

    assert "top_n" not in inspect.signature(shap_to_farmer_explanation).parameters
    assert {group["label"] for group in explanation["groups"]} == set(DRIVER_GROUPS)