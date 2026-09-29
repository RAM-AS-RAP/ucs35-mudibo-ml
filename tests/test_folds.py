"""Tests for shared chronological forecast folds."""

import pandas as pd

from agrisence.training.folds import build_expanding_window_folds


def _supervised_rows(horizon: str) -> pd.DataFrame:
    offset = 1 if horizon == "h1" else 2
    last_origin = pd.Timestamp("2019-10-01") - pd.DateOffset(months=offset)
    origin_months = pd.date_range("2010-01-01", last_origin, freq="MS")
    return pd.DataFrame(
        {
            "market": ["Nairobi"] * len(origin_months),
            "commodity": ["Beans"] * len(origin_months),
            "month": origin_months,
            "target_month": origin_months + pd.DateOffset(months=offset),
            "forecast_horizon": [horizon] * len(origin_months),
        }
    )


def test_folds_split_by_origin_month_with_expected_test_window():
    rows = _supervised_rows("h1")

    result = build_expanding_window_folds(rows)

    test_origins = rows.loc[result["test_mask"], "month"]
    development_origins = rows.loc[result["development_mask"], "month"]
    assert test_origins.min() == pd.Timestamp("2018-11-01")
    assert test_origins.max() == pd.Timestamp("2019-09-01")
    assert development_origins.max() == pd.Timestamp("2018-10-01")
    assert not (result["test_mask"] & result["development_mask"]).any()
    assert len(result["folds"]) == 4


def test_expanding_fold_training_cutoff_respects_forecast_horizon():
    h1 = build_expanding_window_folds(_supervised_rows("h1"))
    h2 = build_expanding_window_folds(_supervised_rows("h2"))

    assert h1["folds"][0]["validation_start"] == pd.Timestamp("2015-01-01")
    assert h1["folds"][0]["train_target_end"] == pd.Timestamp("2015-01-01")
    assert h1["folds"][0]["train_origin_end"] == pd.Timestamp("2014-12-01")
    assert h2["folds"][0]["train_target_end"] == pd.Timestamp("2015-01-01")
    assert h2["folds"][0]["train_origin_end"] == pd.Timestamp("2014-11-01")
    assert h1["folds"][-1]["validation_end"] == pd.Timestamp("2018-10-01")


def test_folds_reject_mixed_horizons_and_missing_target_month():
    rows = _supervised_rows("h1")
    rows.loc[0, "forecast_horizon"] = "h2"
    try:
        build_expanding_window_folds(rows)
    except ValueError as error:
        assert "exactly one forecast_horizon" in str(error)
    else:
        raise AssertionError("mixed horizon table should be rejected")

    rows = _supervised_rows("h1").drop(columns="target_month")
    try:
        build_expanding_window_folds(rows)
    except ValueError as error:
        assert "target_month" in str(error)
    else:
        raise AssertionError("missing target_month should be rejected")
