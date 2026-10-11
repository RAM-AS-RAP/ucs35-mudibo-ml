"""Compare inference features with the saved training panel and supervised table."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrisence.inference.forecast_engine import build_origin_features, prepare_series_panel


ROOT = Path(__file__).resolve().parents[1]
CLEANED_PANEL = ROOT / "results/artifacts/cleaned_monthly_prices_wfp_2010_2019-10.csv"
MODEL_DF_H1 = ROOT / "results/artifacts/feature_engineering/model_df_h1.csv"
pytestmark = pytest.mark.skipif(
    not CLEANED_PANEL.is_file() or not MODEL_DF_H1.is_file(),
    reason="Feature parity requires the cleaned training panel and saved model_df_h1.csv",
)

FEATURE_COLUMNS = [
    "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
    "log_ratio_3m", "log_ratio_12m", "month_sin", "month_cos", "year",
    "market_Nairobi", "market_Kisumu", "market_Mombasa", "market_Nakuru",
    "commodity_Maize", "commodity_Beans",
]


def test_sampled_saved_training_rows_match_truncated_inference_features():
    cleaned = pd.read_csv(CLEANED_PANEL, parse_dates=["month"])
    cleaned["crop"] = cleaned["commodity"].str.lower()
    saved = pd.read_csv(MODEL_DF_H1, parse_dates=["month"])
    sample = saved.groupby(["market", "commodity"], group_keys=False).head(5)
    assert len(sample) >= 20
    assert sample.groupby(["market", "commodity"]).ngroups == 8
    latest_month = cleaned["month"].max()

    for row in sample.itertuples(index=False):
        origin = pd.Timestamp(row.month).date()
        panel = prepare_series_panel(cleaned, row.market, row.commodity.lower(), origin)
        features = build_origin_features(panel, origin, FEATURE_COLUMNS)
        expected = np.asarray([getattr(row, column) for column in FEATURE_COLUMNS], dtype=np.float64)
        np.testing.assert_allclose(features.iloc[0].to_numpy(dtype=np.float64), expected, rtol=0, atol=1e-9)

        full_panel = prepare_series_panel(cleaned, row.market, row.commodity.lower(), latest_month)
        full_features = build_origin_features(full_panel, origin, FEATURE_COLUMNS)
        np.testing.assert_allclose(
            full_features.iloc[0].to_numpy(dtype=np.float64),
            features.iloc[0].to_numpy(dtype=np.float64),
            rtol=0,
            atol=1e-9,
        )