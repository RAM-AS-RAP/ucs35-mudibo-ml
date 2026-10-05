"""Linear regression baseline trainer."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_linear_regression(X_train: Any, y_train: Any) -> Pipeline:
    """Fit a scaled linear-regression pipeline on one training fold."""
    model = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())],
        memory=None,
    )
    return model.fit(X_train, y_train)


def train_linear_regression(X_train: Any, y_train: Any) -> Pipeline:
    """Backward-compatible alias for ``fit_linear_regression``."""
    return fit_linear_regression(X_train, y_train)
