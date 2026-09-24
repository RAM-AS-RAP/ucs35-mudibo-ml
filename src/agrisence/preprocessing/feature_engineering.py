"""Feature engineering utilities for crop price modeling.

This module creates lag, rolling, and seasonal features for a multivariate forecasting task.
"""

from __future__ import annotations

import pandas as pd


def add_lag_features() -> pd.DataFrame:
    """Add lagged values for the target variable and related price features."""
    # TODO: create lag features such as 1-month and 2-month lags.
    return pd.DataFrame()


def add_rolling_averages() -> pd.DataFrame:
    """Add rolling mean and moving average features to the feature set."""
    # TODO: compute rolling averages for recent price behavior.
    return pd.DataFrame()


def add_seasonal_indicators() -> pd.DataFrame:
    """Add month and seasonal indicator columns for recurring climate/market patterns."""
    # TODO: add cyclical seasonal features based on month or season.
    return pd.DataFrame()


def build_feature_set(df: pd.DataFrame, include_climate: bool = True) -> pd.DataFrame:
    """Compose the final feature set used in training and evaluation.

    The include_climate flag exists to support the climate-variable ablation experiment.
    """
    # TODO: assemble the feature set, optionally add climate variables, and return df.
    return df.copy()
