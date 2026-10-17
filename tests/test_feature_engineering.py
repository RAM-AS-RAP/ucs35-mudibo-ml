"""Regression tests for monthly price feature and target alignment."""

import numpy as np
import pandas as pd

from agrisence.preprocessing.feature_engineering import (
    apply_calendar_distance_guard,
    build_features,
    build_supervised_table,
    build_targets,
    filter_observed_eligible,
)


def _panel(months: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": "Nairobi",
            "commodity": "Beans",
            "month": months,
            "avg_price_per_kg": np.arange(20, 20 + len(months), dtype=float),
            "impute_method": "observed",
        }
    )


def test_build_features_creates_log_returns_ratios_calendar_and_identity():
    panel = _panel(pd.date_range("2010-01-01", periods=16, freq="MS"))

    result = build_features(panel)

    march = result.loc[result["month"].eq(pd.Timestamp("2010-03-01"))].iloc[0]
    expected_return = np.log(21.0) - np.log(20.0)
    assert np.isclose(march["log_return_lag_1"], expected_return)
    assert np.isnan(result.loc[result["month"].eq("2010-01-01"), "log_ratio_12m"]).all()
    assert result["month_sin"].iloc[0] == 0
    assert result["month_cos"].iloc[0] == 1
    assert result["market_Nairobi"].eq(1).all()
    assert result["commodity_Beans"].eq(1).all()


def test_build_targets_aligns_exact_calendar_months_across_missing_row():
    months = pd.date_range("2010-01-01", periods=8, freq="MS").delete(2)
    panel = build_targets(_panel(months))

    february = panel.loc[panel["month"].eq(pd.Timestamp("2010-02-01"))].iloc[0]
    april = panel.loc[panel["month"].eq(pd.Timestamp("2010-04-01"))].iloc[0]
    assert pd.isna(february["target_h1"])
    assert np.isclose(april["target_h1"], np.log(23.0) - np.log(22.0))
    assert "target_h2" not in panel.columns


def test_calendar_guard_invalidates_features_crossing_missing_month():
    months = pd.date_range("2010-01-01", periods=16, freq="MS").delete(5)
    panel = apply_calendar_distance_guard(
        build_targets(build_features(_panel(months)))
    )

    july = panel.loc[panel["month"].eq(pd.Timestamp("2010-07-01"))].iloc[0]
    assert pd.isna(july["log_return_lag_1"])
    assert pd.isna(july["log_ratio_3m"])
    assert not july["consecutive_observation"]


def test_observed_eligibility_requires_observed_origin_and_target(capsys):
    panel = build_targets(_panel(pd.date_range("2010-01-01", periods=6, freq="MS")))
    panel.loc[panel["month"].eq("2010-02-01"), "impute_method"] = "short_gap_ffill"
    panel = build_targets(panel)

    eligible = filter_observed_eligible(panel, "h1")

    assert eligible["impute_method"].eq("observed").all()
    assert eligible["target_method_h1"].eq("observed").all()
    assert "rows excluded" in capsys.readouterr().out


def test_supervised_table_separates_endpoint_and_feature_exclusions(capsys):
    panel = _panel(pd.date_range("2009-01-01", periods=36, freq="MS"))
    panel.loc[panel["month"].eq("2010-06-01"), "impute_method"] = "short_gap_ffill"
    panel = apply_calendar_distance_guard(
        build_targets(build_features(panel))
    )

    supervised = build_supervised_table(
        panel,
        "h1",
        window_start="2010-01-01",
        window_end="2011-12-01",
    )

    output = capsys.readouterr().out
    assert "rows excluded for non-observed/missing endpoints" in output
    assert "excluded for missing features" in output
    assert supervised["impute_method"].eq("observed").all()
    assert supervised["target_month"].eq(supervised["month"] + pd.DateOffset(months=1)).all()
    assert supervised["target_method_h1"].eq("observed").all()
    assert supervised["log_ratio_12m"].notna().all()
