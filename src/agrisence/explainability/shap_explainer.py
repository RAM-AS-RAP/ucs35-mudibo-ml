"""Exact SHAP explanations for the Random Forest/XGBoost/Ridge stack."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DRIVER_GROUPS = {
    "Recent price momentum": [
        "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
    ],
    "Price compared with its recent average": [
        "log_ratio_3m", "log_ratio_12m",
    ],
    "Time of year": ["month_sin", "month_cos"],
}
BASELINE_GROUPS = {
    "Long-run trend": ["year"],
    "Market and crop": "PREFIX:market_,commodity_",
}
TYPICAL_CHANGE_LABEL = "Typical change for this market and crop"


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


def _validate_grouped_shap_inputs(
    shap_result: dict,
    row_index: int,
    origin_price: float,
    neutral_threshold: float,
) -> tuple[list[str], np.ndarray, float, float, float]:
    if not isinstance(row_index, int) or isinstance(row_index, bool):
        raise TypeError("row_index must be an integer row position")
    if not np.isfinite(origin_price) or origin_price <= 0:
        raise ValueError("origin_price must be finite and positive")
    if not np.isfinite(neutral_threshold) or neutral_threshold < 0:
        raise ValueError("neutral_threshold must be finite and nonnegative")

    values = np.asarray(shap_result["shap_values"], dtype=float)
    feature_names = list(shap_result["feature_names"])
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP values must be a 2D array aligned with feature_names")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values must be finite")
    if row_index < 0 or row_index >= values.shape[0]:
        raise IndexError("row_index is outside the explained rows")
    if len(set(feature_names)) != len(feature_names):
        duplicates = sorted({name for name in feature_names if feature_names.count(name) > 1})
        raise ValueError(f"SHAP feature names are duplicated: {duplicates}")
    base_values = np.asarray(shap_result["base_value"], dtype=float).reshape(-1)
    if base_values.size != 1 or not np.isfinite(base_values[0]):
        raise ValueError("base_value must be one finite scalar")
    return feature_names, values[row_index], float(base_values[0]), float(origin_price), float(neutral_threshold)


def _feature_group_members(group_definition: list[str] | str, feature_names: list[str]) -> list[str]:
    if isinstance(group_definition, str) and group_definition.startswith("PREFIX:"):
        prefixes = tuple(group_definition.removeprefix("PREFIX:").split(","))
        return [name for name in feature_names if name.startswith(prefixes)]
    feature_name_set = set(feature_names)
    return [name for name in group_definition if name in feature_name_set]


def _resolve_feature_groups(
    feature_names: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    assignments: dict[str, list[str]] = {}

    def resolve(definitions: dict[str, list[str] | str]) -> dict[str, list[str]]:
        resolved = {}
        for label, definition in definitions.items():
            members = _feature_group_members(definition, feature_names)
            if members:
                resolved[label] = members
                for name in members:
                    assignments.setdefault(name, []).append(label)
        return resolved

    driver_groups = resolve(DRIVER_GROUPS)
    baseline_groups = resolve(BASELINE_GROUPS)
    unassigned = [name for name in feature_names if name not in assignments]
    multiply_assigned = sorted(name for name, labels in assignments.items() if len(labels) != 1)
    if unassigned or multiply_assigned:
        errors = []
        if unassigned:
            errors.append(f"unassigned features: {unassigned}")
        if multiply_assigned:
            errors.append(f"multiply assigned features: {multiply_assigned}")
        raise ValueError("Invalid SHAP feature grouping; " + "; ".join(errors))
    return driver_groups, baseline_groups


def _direction_for_contribution(log_contribution: float, neutral_threshold: float) -> str:
    if log_contribution > neutral_threshold:
        return "increases"
    if log_contribution < -neutral_threshold:
        return "decreases"
    return "neutral"


def shap_to_farmer_explanation(
    shap_result: dict,
    row_index: int,
    origin_price: float,
    neutral_threshold: float = 0.001,
    model_prediction: float | None = None,
) -> dict[str, Any]:
    """Separate the typical market change from a forecast's SHAP drivers.

    Log contributions reconcile exactly with the predicted log change before
    presentation rounding. Percentages are first-order approximations and do not
    sum to the headline percentage. The typical change includes the model's average
    forecast plus the fixed effects of year, market, and crop. SHAP describes how
    this forecast differs from the model's average forecast, not the causes of
    price movement.
    """
    feature_names, row_values, base_value, origin_price, neutral_threshold = (
        _validate_grouped_shap_inputs(shap_result, row_index, origin_price, neutral_threshold)
    )
    driver_groups, baseline_groups = _resolve_feature_groups(feature_names)
    feature_indices = {name: index for index, name in enumerate(feature_names)}
    driver_contributions = {
        label: float(sum(row_values[feature_indices[name]] for name in members))
        for label, members in driver_groups.items()
    }
    baseline_contributions = {
        label: float(sum(row_values[feature_indices[name]] for name in members))
        for label, members in baseline_groups.items()
    }
    typical_log = base_value + sum(baseline_contributions.values())
    predicted_log_change = typical_log + sum(driver_contributions.values())
    shap_reconstruction = base_value + float(row_values.sum())
    if not np.isclose(predicted_log_change, shap_reconstruction, rtol=0.0, atol=1e-6):
        raise AssertionError(
            "Driver and baseline SHAP contributions do not reconcile with the full feature contributions: "
            f"grouped={predicted_log_change}, full={shap_reconstruction}"
        )
    if model_prediction is not None and not np.isclose(
        predicted_log_change, float(model_prediction), rtol=0.0, atol=1e-6,
    ):
        raise AssertionError(
            "Grouped SHAP forecast does not reconcile with model_prediction: "
            f"grouped={predicted_log_change}, model_prediction={model_prediction}"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        predicted_price = float(origin_price * np.exp(predicted_log_change))
        expected_change_pct = float((np.exp(predicted_log_change) - 1.0) * 100.0)
        typical_change_pct = float((np.exp(typical_log) - 1.0) * 100.0)
    if not np.isfinite(predicted_price) or not np.isfinite(expected_change_pct):
        raise ValueError("Reconstructed forecast price and percent change must be finite")
    if not np.isfinite(typical_change_pct):
        raise ValueError("Typical change percent must be finite")

    groups = []
    for label, log_contribution in driver_contributions.items():
        with np.errstate(over="ignore", invalid="ignore"):
            pct_contribution = float((np.exp(log_contribution) - 1.0) * 100.0)
        if not np.isfinite(pct_contribution):
            raise ValueError(f"Percent contribution for {label!r} must be finite")
        groups.append({
            "label": label,
            "log_contribution": log_contribution,
            "pct_contribution": pct_contribution,
            "direction": _direction_for_contribution(log_contribution, neutral_threshold),
        })
    groups.sort(key=lambda group: abs(group["log_contribution"]), reverse=True)

    return {
        "origin_price": round(float(origin_price), 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_log_change": round(float(predicted_log_change), 4),
        "expected_change_pct": round(expected_change_pct, 2),
        "base_value": base_value,
        "typical_change": {
            "label": TYPICAL_CHANGE_LABEL,
            "log_value": round(float(typical_log), 4),
            "pct_value": round(typical_change_pct, 2),
        },
        "groups": [
            {
                **group,
                "log_contribution": round(group["log_contribution"], 4),
                "pct_contribution": round(group["pct_contribution"], 2),
            }
            for group in groups
        ],
    }