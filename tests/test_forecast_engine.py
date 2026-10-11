"""Pure synthetic forecast-engine checks; no fitted model or data services."""

from datetime import date
import warnings

import numpy as np
import pandas as pd
import pytest

from agrisence.inference import forecast_engine
from agrisence.inference.forecast_engine import FeatureError, build_origin_features, predict_and_explain, prepare_series_panel


FEATURE_COLUMNS = [
    "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
    "log_ratio_3m", "log_ratio_12m", "month_sin", "month_cos", "year",
    "market_Nairobi", "market_Kisumu", "market_Mombasa", "market_Nakuru",
    "commodity_Maize", "commodity_Beans",
]


def test_prepare_panel_reindexes_gaps_and_excludes_later_months():
    prices = pd.DataFrame({
        "market": ["Nairobi", "Nairobi", "Nairobi"],
        "crop": ["maize"] * 3,
        "month": ["2025-01-01", "2025-03-01", "2025-04-01"],
        "avg_price_per_kg": [100.0, 110.0, 120.0],
    })
    panel = prepare_series_panel(prices, "Nairobi", "maize", date(2025, 3, 15))
    assert panel["month"].tolist() == list(pd.date_range("2025-01-01", "2025-03-01", freq="MS"))
    assert pd.isna(panel.loc[1, "avg_price_per_kg"])
    assert panel["avg_price_per_kg"].notna().sum() == 2


def test_build_origin_features_names_missing_columns_on_history_error():
    prices = pd.DataFrame({
        "market": ["Nairobi"] * 3, "crop": ["maize"] * 3,
        "month": pd.date_range("2025-01-01", periods=3, freq="MS"),
        "avg_price_per_kg": [100.0, 101.0, 102.0],
    })
    panel = prepare_series_panel(prices, "Nairobi", "maize", date(2025, 3, 1))
    with pytest.raises(FeatureError, match="Nairobi/Maize origin 2025-03-01.*log_return_lag_2"):
        build_origin_features(panel, date(2025, 3, 1), FEATURE_COLUMNS)


class _ConstantPredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _models(prediction):
    class RidgeFixture:
        def __init__(self):
            self.prediction = prediction

        def predict(self, values):
            return np.full(len(values), self.prediction, dtype=float)

    return {
        "rf_model": _ConstantPredictor(0.0),
        "xgb_model": _ConstantPredictor(0.0),
        "ridge_meta": RidgeFixture(),
    }


def test_prediction_reconciles_and_returns_full_grouped_explanation(monkeypatch):
    monkeypatch.setattr(forecast_engine, "compute_stack_shap_values", lambda models, X: {
        "shap_values": np.zeros((len(X), len(FEATURE_COLUMNS))),
        "base_value": 0.02,
        "feature_names": FEATURE_COLUMNS,
    })
    result = predict_and_explain(_models(0.02), pd.DataFrame([[0.0] * 14], columns=FEATURE_COLUMNS), [100.0])
    assert set(result[0]) == {"predicted_log_change", "predicted_price", "shap_values"}
    assert result[0]["predicted_log_change"] == 0.02
    assert result[0]["shap_values"]["origin_price"] == 100.0
    assert {group["label"] for group in result[0]["shap_values"]["groups"]} == {
        "Recent price momentum", "Price compared with its recent average", "Time of year",
    }


def _patch_shap(monkeypatch):
    monkeypatch.setattr(forecast_engine, "compute_stack_shap_values", lambda models, X: {
        "shap_values": np.zeros((len(X), len(FEATURE_COLUMNS))),
        "base_value": models["ridge_meta"].prediction,
        "feature_names": FEATURE_COLUMNS,
    })


def test_large_forecast_warns(monkeypatch):
    _patch_shap(monkeypatch)
    X = pd.DataFrame([[0.0] * 14], columns=FEATURE_COLUMNS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predict_and_explain(_models(0.31), X, [100.0])
    assert any("Large predicted log change" in str(item.message) for item in caught)


def test_excessive_forecast_change_fails(monkeypatch):
    _patch_shap(monkeypatch)
    X = pd.DataFrame([[0.0] * 14], columns=FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="exceeds 0.7"):
        predict_and_explain(_models(0.71), X, [100.0])