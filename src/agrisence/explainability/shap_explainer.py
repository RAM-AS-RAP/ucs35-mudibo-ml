"""Exact SHAP explanations for the Random Forest/XGBoost/Ridge stack."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _single_output_shap_values(values: Any, rows: int, features: int, model_name: str) -> np.ndarray:
    """Normalize the supported single-output TreeExplainer result to 2D."""
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"{model_name} SHAP output must have exactly one model output")
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.shape != (rows, features):
        raise ValueError(
            f"{model_name} SHAP output has shape {values.shape}; expected {(rows, features)}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{model_name} SHAP output must be finite")
    return values


def _single_output_base_value(value: Any, model_name: str) -> float:
    """Normalize a single-output TreeExplainer expected value to a scalar."""
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"{model_name} expected value must be one finite scalar")
    return float(values[0])


def compute_stack_shap_values(models: dict, X: pd.DataFrame) -> dict:
    """Compute exact additive SHAP values for the Ridge-weighted tree stack.

    The stack prediction is the linear combination of the Random Forest and XGBoost
    predictions plus Ridge's intercept. By Shapley additivity, applying the same Ridge
    coefficients to each base model's SHAP values and expected value exactly decomposes
    the stacked prediction. This is valid only for the linear Ridge meta-learner; it is
    not valid for a nonlinear meta-model.
    """
    required_models = {"rf_model", "xgb_model", "ridge_meta"}
    missing_models = required_models.difference(models)
    if missing_models:
        raise ValueError(f"Model bundle is missing required estimators: {sorted(missing_models)}")
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a nonempty pandas DataFrame")
    if not X.columns.is_unique:
        raise ValueError("X feature names must be unique")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X must contain only finite numeric values")
    if "feature_cols" in models and list(X.columns) != list(models["feature_cols"]):
        raise ValueError("X columns and order must match the saved model feature_cols")

    ridge = models["ridge_meta"]
    if not isinstance(ridge, Ridge):
        raise TypeError("Exact stack SHAP decomposition requires a fitted Ridge meta-model")
    coefficients = np.asarray(ridge.coef_, dtype=float).reshape(-1)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
        raise ValueError("Ridge must have exactly two finite coefficients for RF and XGBoost")
    intercept = np.asarray(ridge.intercept_, dtype=float).reshape(-1)
    if intercept.size != 1 or not np.isfinite(intercept[0]):
        raise ValueError("Ridge must have one finite intercept")

    import shap

    rf_explainer = shap.TreeExplainer(models["rf_model"])
    xgb_explainer = shap.TreeExplainer(models["xgb_model"])
    rf_values = _single_output_shap_values(
        rf_explainer.shap_values(X), len(X), X.shape[1], "Random Forest",
    )
    xgb_values = _single_output_shap_values(
        xgb_explainer.shap_values(X), len(X), X.shape[1], "XGBoost",
    )
    rf_base = _single_output_base_value(rf_explainer.expected_value, "Random Forest")
    xgb_base = _single_output_base_value(xgb_explainer.expected_value, "XGBoost")

    combined_values = coefficients[0] * rf_values + coefficients[1] * xgb_values
    combined_base = float(intercept[0] + coefficients[0] * rf_base + coefficients[1] * xgb_base)
    return {
        "shap_values": combined_values,
        "base_value": combined_base,
        "feature_names": list(X.columns),
    }


def shap_to_farmer_explanation(
    shap_result: dict,
    row_index: int,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return the largest absolute feature contributions for one explained row."""
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 0:
        raise ValueError("top_n must be a nonnegative integer")
    values = np.asarray(shap_result["shap_values"], dtype=float)
    feature_names = list(shap_result["feature_names"])
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP values must be a 2D array aligned with feature_names")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values must be finite")
    if not isinstance(row_index, int) or isinstance(row_index, bool):
        raise TypeError("row_index must be an integer row position")
    if row_index < 0 or row_index >= values.shape[0]:
        raise IndexError("row_index is outside the explained rows")

    row_values = values[row_index]
    top_indices = np.argsort(-np.abs(row_values), kind="stable")[:top_n]
    return [
        {
            "feature": feature_names[index],
            "contribution": float(row_values[index]),
            "direction": "increases" if row_values[index] > 0 else "decreases",
        }
        for index in top_indices
    ]