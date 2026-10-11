"""Actual model smoke test; requires saved verified models and compatible runtimes."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrisence.inference.forecast_engine import build_origin_features, predict_and_explain, prepare_series_panel
from agrisence.inference.model_loader import CHECKSUM_PATH, MODEL_PATHS, ModelLoadError, load_all_models
from agrisence.pipelines.forecast_origins import plan_origins, target_month


pytestmark = pytest.mark.models
PORTABLE_MODEL_PATHS = [path for paths in MODEL_PATHS.values() for path in paths.values()]
MISSING_ASSETS = [str(path) for path in (*PORTABLE_MODEL_PATHS, CHECKSUM_PATH) if not Path(path).is_file()]
if MISSING_ASSETS:
    pytest.skip(f"Model smoke test requires committed model files/checksums: {MISSING_ASSETS}", allow_module_level=True)


def test_synthetic_random_walk_forecasts_both_horizons():
    months = pd.date_range("2017-01-01", periods=14, freq="MS")
    prices = 100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0.005, 0.015, len(months))))
    prices_df = pd.DataFrame({
        "market": "Nairobi", "crop": "maize", "month": months,
        "avg_price_per_kg": prices,
    })
    today = months[-1].date()
    origin_plan = plan_origins([month.date() for month in months], {}, today)
    planned_origins = [origin_plan["live_origin"], *origin_plan["backfill_origins"]]
    assert origin_plan["eligible"]
    assert len(planned_origins) == 2

    try:
        models = load_all_models()
    except ModelLoadError as exception:
        pytest.skip(f"Verified model files are unavailable for smoke inference: {exception}")
    for horizon_key, horizon_name in (("h1", "one_month"), ("h2", "two_month")):
        features = []
        origin_prices = []
        for origin in planned_origins:
            panel = prepare_series_panel(prices_df, "Nairobi", "maize", origin)
            features.append(build_origin_features(panel, origin, models[horizon_key]["feature_cols"]))
            origin_prices.append(float(prices_df.loc[prices_df["month"].dt.date.eq(origin), "avg_price_per_kg"].iloc[0]))
        results = predict_and_explain(models[horizon_key], pd.concat(features, ignore_index=True), origin_prices)
        assert len(results) == 2
        for origin, result, origin_price in zip(planned_origins, results, origin_prices, strict=True):
            assert set(result) == {"predicted_log_change", "predicted_price", "shap_values"}
            explanation = result["shap_values"]
            assert explanation["origin_price"] == round(origin_price, 2)
            assert explanation["predicted_log_change"] == result["predicted_log_change"]
            displayed_reconstruction = (
                sum(group["log_contribution"] for group in explanation["groups"])
                + explanation["typical_change"]["log_value"]
            )
            assert displayed_reconstruction == pytest.approx(
                result["predicted_log_change"], rel=0, abs=3e-4,
            )
            assert target_month(origin, horizon_name) > origin