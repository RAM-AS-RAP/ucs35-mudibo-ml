"""SARIMA baseline trainer."""

from __future__ import annotations

from typing import Any


def train_sarima(series: Any, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> Any:
    """Train a SARIMA baseline model on a univariate time series."""
    # TODO: fit a statsmodels SARIMAX model and return the fitted result.
    return None
